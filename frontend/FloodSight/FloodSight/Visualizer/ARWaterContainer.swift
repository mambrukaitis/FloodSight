//
//  ARWaterContainer.swift
//  FloodSight
//
//  Created by Matas Ambrukaitis on 10/23/26.
//

import SwiftUI
import RealityKit
import ARKit
import Combine
import Vision
import CoreML
import CoreVideo
import ImageIO

struct ARWaterContainer: UIViewRepresentable {
    @Binding var waterLevel: Float
    func makeUIView(context: Context) -> CustomARView { return CustomARView(frame: .zero) }
    func updateUIView(_ uiView: CustomARView, context: Context) { uiView.updateWaterHeight(to: waterLevel) }
}

// MARK: - Optional Far-Field Depth Helpers (safe off the main thread)

private nonisolated struct DepthGrid {
    let values: [Float]
    let width: Int
    let height: Int
}

private nonisolated struct DepthFrameContext: @unchecked Sendable {
    let pixelBuffer: CVPixelBuffer
    let camTransform: simd_float4x4
    let intrinsics: simd_float3x3
    let imageSize: CGSize
    let orientation: CGImagePropertyOrientation
    let floorY: Float
    let waterY: Float
    let nearTrustRange: Float
    let lidarSamples: [SIMD3<Float>]
}

private nonisolated struct OccluderResult: @unchecked Sendable {
    let positions: [SIMD3<Float>]
    let indices: [UInt32]
}

private nonisolated enum DepthOutcome: @unchecked Sendable {
    case occluder(OccluderResult)
    case failed(String)
}

private nonisolated enum FarFieldDepth {

    static let lidarTrustRange: Float = 4.5
    static let maxRange: Float = 250
    static let occluderDepthBias: Float = 1.10
    static let gridColumns = 64
    static let gridRows = 48

    static func run(model: VNCoreMLModel, ctx: DepthFrameContext) -> DepthOutcome {
        let request = VNCoreMLRequest(model: model)
        request.imageCropAndScaleOption = .scaleFill
        let handler = VNImageRequestHandler(cvPixelBuffer: ctx.pixelBuffer,
                                            orientation: ctx.orientation,
                                            options: [:])
        do { try handler.perform([request]) } catch { return .failed("Vision request failed: \(error)") }

        guard let obs = request.results?.first as? VNPixelBufferObservation,
              let grid = readGrid(obs.pixelBuffer) else {
            return .failed("model output was not a readable depth image")
        }
        guard let fit = calibrate(grid: grid, ctx: ctx) else {
            return .failed("could not calibrate depth")
        }
        guard let occluder = buildOccluder(grid: grid, ctx: ctx, fit: fit) else {
            return .failed("nothing beyond LiDAR range in view")
        }
        return .occluder(occluder)
    }

    static func readGrid(_ pb: CVPixelBuffer) -> DepthGrid? {
        CVPixelBufferLockBaseAddress(pb, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(pb) else { return nil }
        let w = CVPixelBufferGetWidth(pb)
        let h = CVPixelBufferGetHeight(pb)
        let rowBytes = CVPixelBufferGetBytesPerRow(pb)
        var out = [Float](repeating: 0, count: w * h)

        switch CVPixelBufferGetPixelFormatType(pb) {
        case kCVPixelFormatType_OneComponent16Half, kCVPixelFormatType_DepthFloat16, kCVPixelFormatType_DisparityFloat16:
            for y in 0..<h {
                let row = base.advanced(by: y * rowBytes).assumingMemoryBound(to: Float16.self)
                for x in 0..<w { out[y * w + x] = Float(row[x]) }
            }
        case kCVPixelFormatType_OneComponent32Float, kCVPixelFormatType_DepthFloat32, kCVPixelFormatType_DisparityFloat32:
            for y in 0..<h {
                let row = base.advanced(by: y * rowBytes).assumingMemoryBound(to: Float32.self)
                for x in 0..<w { out[y * w + x] = row[x] }
            }
        case kCVPixelFormatType_OneComponent8:
            for y in 0..<h {
                let row = base.advanced(by: y * rowBytes).assumingMemoryBound(to: UInt8.self)
                for x in 0..<w { out[y * w + x] = Float(row[x]) / 255.0 }
            }
        default:
            return nil
        }
        return DepthGrid(values: out, width: w, height: h)
    }

    static func uprightUV(_ u: Float, _ v: Float, _ W: Float, _ H: Float,
                          _ o: CGImagePropertyOrientation) -> SIMD2<Float> {
        let nu = u / W
        let nv = v / H
        switch o {
        case .right: return SIMD2<Float>(1 - nv, nu)
        case .left:  return SIMD2<Float>(nv, 1 - nu)
        case .down:  return SIMD2<Float>(1 - nu, 1 - nv)
        default:     return SIMD2<Float>(nu, nv)
        }
    }

    static func sample(_ g: DepthGrid, _ uv: SIMD2<Float>) -> Float {
        let fx = min(max(uv.x * Float(g.width) - 0.5, 0), Float(g.width - 1))
        let fy = min(max(uv.y * Float(g.height) - 0.5, 0), Float(g.height - 1))
        let x0 = Int(fx), y0 = Int(fy)
        let x1 = min(x0 + 1, g.width - 1), y1 = min(y0 + 1, g.height - 1)
        let tx = fx - Float(x0), ty = fy - Float(y0)
        let top = g.values[y0 * g.width + x0] * (1 - tx) + g.values[y0 * g.width + x1] * tx
        let bot = g.values[y1 * g.width + x0] * (1 - tx) + g.values[y1 * g.width + x1] * tx
        return top * (1 - ty) + bot * ty
    }

    static func calibrate(grid: DepthGrid, ctx: DepthFrameContext) -> (a: Float, b: Float)? {
        let W = Float(ctx.imageSize.width)
        let H = Float(ctx.imageSize.height)
        var inv = [Float]()
        var model = [Float]()

        for s in ctx.lidarSamples {
            inv.append(1.0 / s.z)
            model.append(sample(grid, uprightUV(s.x, s.y, W, H, ctx.orientation)))
        }

        let k = ctx.intrinsics
        let heightAboveFloor = ctx.camTransform.columns.3.y - ctx.floorY
        if heightAboveFloor > 0.4 {
            for v in Swift.stride(from: Float(0), to: H, by: 48) {
                for u in Swift.stride(from: Float(0), to: W, by: 48) {
                    let cx = (u - k[2][0]) / k[0][0]
                    let cy = (v - k[2][1]) / k[1][1]
                    let ray = ctx.camTransform * SIMD4<Float>(cx, -cy, -1, 0)
                    guard ray.y < -0.1 else { continue }
                    let z = -heightAboveFloor / ray.y
                    guard z > 1.0 && z < 12.0 else { continue }
                    inv.append(1.0 / z)
                    model.append(sample(grid, uprightUV(u, v, W, H, ctx.orientation)))
                }
            }
        }

        guard inv.count >= 80 else { return nil }

        var keep = Array(0..<inv.count).filter { model[$0].isFinite }
        var a: Float = 0
        var b: Float = 0
        for _ in 0..<3 {
            guard keep.count >= 60 else { return nil }
            var sx = 0.0, sy = 0.0, sxx = 0.0, sxy = 0.0
            for i in keep {
                let x = Double(inv[i]), y = Double(model[i])
                sx += x; sy += y; sxx += x * x; sxy += x * y
            }
            let n = Double(keep.count)
            let den = n * sxx - sx * sx
            guard abs(den) > 1e-12 else { return nil }
            let slope = (n * sxy - sx * sy) / den
            a = Float(slope)
            b = Float((sy - slope * sx) / n)

            let resid = keep.map { abs(model[$0] - (a * inv[$0] + b)) }
            let median = resid.sorted()[resid.count / 2]
            let cut = max(median * 3.0, 1e-6)
            keep = keep.filter { abs(model[$0] - (a * inv[$0] + b)) <= cut }
        }
        guard keep.count >= 60, Float(keep.count) > 0.35 * Float(inv.count), a > 0 else { return nil }
        return (a, b)
    }

    static func unproject(_ u: Float, _ v: Float, _ z: Float, _ ctx: DepthFrameContext) -> SIMD3<Float> {
        let k = ctx.intrinsics
        let x = (u - k[2][0]) / k[0][0] * z
        let y = (v - k[2][1]) / k[1][1] * z
        let p = ctx.camTransform * SIMD4<Float>(x, -y, -z, 1)
        return SIMD3<Float>(p.x, p.y, p.z)
    }

    static func buildOccluder(grid: DepthGrid, ctx: DepthFrameContext,
                              fit: (a: Float, b: Float)) -> OccluderResult? {
        let W = Float(ctx.imageSize.width)
        let H = Float(ctx.imageSize.height)
        let cols = gridColumns
        let rows = gridRows
        let rowLen = cols + 1

        var positions = [SIMD3<Float>]()
        var depths = [Float]()
        positions.reserveCapacity(rowLen * (rows + 1))
        depths.reserveCapacity(rowLen * (rows + 1))

        for j in 0...rows {
            for i in 0...cols {
                let u = Float(i) / Float(cols) * W
                let v = Float(j) / Float(rows) * H
                let d = sample(grid, uprightUV(u, v, W, H, ctx.orientation))
                let denom = d - fit.b
                var z = Float.infinity
                if denom > fit.a / maxRange {
                    z = fit.a / denom * occluderDepthBias
                }
                depths.append(z)
                positions.append(z.isFinite ? unproject(u, v, z, ctx) : .zero)
            }
        }

        var indices = [UInt32]()
        indices.reserveCapacity(rows * cols * 6)

        func addTriangle(_ p: Int, _ q: Int, _ r: Int) {
            let zp = depths[p], zq = depths[q], zr = depths[r]
            guard zp.isFinite, zq.isFinite, zr.isFinite else { return }
            guard max(zp, zq, zr) > ctx.nearTrustRange else { return }
            
            // Restored Fix: Only occlude with surfaces that actually rise ABOVE the water plane.
            // This prevents the physical ground from generating an occluder that hides the water on top of it.
            let py = positions[p].y, qy = positions[q].y, ry = positions[r].y
            guard max(py, qy, ry) > ctx.waterY + 0.12 else { return }
            
            indices.append(UInt32(p)); indices.append(UInt32(q)); indices.append(UInt32(r))
        }

        for j in 0..<rows {
            for i in 0..<cols {
                let a = j * rowLen + i
                let b = a + 1
                let c = a + rowLen
                let d = c + 1
                addTriangle(a, c, b)
                addTriangle(b, c, d)
            }
        }

        guard !indices.isEmpty else { return nil }
        return OccluderResult(positions: positions, indices: indices)
    }
}

// MARK: - CustomARView

class CustomARView: ARView, ARSessionDelegate {

    private var worldAnchor: AnchorEntity!
    private var waterEntity: ModelEntity?
    private var subsurfaceMurkEntity: ModelEntity?
    private var standardWaterMaterial = PhysicallyBasedMaterial()
    private var updateSubscription: Cancellable?

    private var lowestFloorY: Float = -1.5
    private var hasDetectedFloor: Bool = false
    private var currentWaterDepth: Float = 0.5

    // Camera-centred polar mesh extending all the way out to 150m (~500 ft) unconditionally
    private let ringCount: Int = 64
    private let segmentCount: Int = 72
    private let innerRadius: Float = 0.5
    private let farRadius: Float = 150.0
    private let foamExtent: Float = 28.0
    private let murkDiameter: Float = 300.0 // Full coverage out to 150m radius
    private var basePositions: [SIMD3<Float>] = []
    private var baseRadii: [Float] = []
    private var baseUVs: [SIMD2<Float>] = []
    private var triangleIndices: [UInt32] = []

    private var elapsedTime: Float = 0.0
    private var frameCounter: Int = 0

    // Spatial caches for LiDAR + Vertical Wall Plane Occlusion
    private var scannedMeshVertices: [UUID: [SIMD3<Float>]] = [:]
    private var planeOcclusionEntities: [UUID: ModelEntity] = [:]
    private var elevatedZones: [UUID: (center: SIMD3<Float>, radius: Float, topY: Float)] = [:]
    private var lastShorelineUpdateTime: TimeInterval = 0

    // Optional Core ML far-field depth
    private var depthModel: VNCoreMLModel?
    private let depthQueue = DispatchQueue(label: "floodsight.farfield.depth", qos: .userInitiated)
    private let depthInterval: TimeInterval = 0.15
    private var lastDepthTimestamp: TimeInterval = 0
    private var isInferring = false
    private var hasLiDAR = false
    private var depthOccluderEntity: ModelEntity?
    private var lastDepthIssue: String = ""

    required init(frame frameRect: CGRect) {
        super.init(frame: frameRect)
        commonInit()
    }

    required init?(coder decoder: NSCoder) {
        super.init(coder: decoder)
        commonInit()
    }

    private func commonInit() {
        setupARAndStableOcclusion()
        setupLighting()
        buildBaseGridData()
        setupSwampWaterSurface()
        loadDepthModel()
        startSimulationLoop()
    }

    private func setupARAndStableOcclusion() {
        self.session.delegate = self
        let config = ARWorldTrackingConfiguration()
        config.planeDetection = [.horizontal, .vertical]
        config.environmentTexturing = .automatic

        if ARWorldTrackingConfiguration.supportsSceneReconstruction(.meshWithClassification) {
            config.sceneReconstruction = .meshWithClassification
            hasLiDAR = true
        } else if ARWorldTrackingConfiguration.supportsSceneReconstruction(.mesh) {
            config.sceneReconstruction = .mesh
            hasLiDAR = true
        }

        var semantics: ARConfiguration.FrameSemantics = []
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.personSegmentationWithDepth) {
            semantics.insert(.personSegmentationWithDepth)
        }
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.smoothedSceneDepth) {
            semantics.insert(.smoothedSceneDepth)
        } else if ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) {
            semantics.insert(.sceneDepth)
        }
        config.frameSemantics = semantics

        self.environment.sceneUnderstanding.options = [.occlusion]
        self.session.run(config)
    }

    private func setupLighting() {
        worldAnchor = AnchorEntity(world: .zero)
        self.scene.addAnchor(worldAnchor)

        let sunLight = DirectionalLight()
        sunLight.light.color = UIColor(red: 0.95, green: 0.96, blue: 0.88, alpha: 1.0)
        sunLight.light.intensity = 2500
        sunLight.look(at: .zero, from: SIMD3<Float>(2, 9, 3), relativeTo: nil)
        worldAnchor.addChild(sunLight)

        let ambientSky = DirectionalLight()
        ambientSky.light.color = UIColor(red: 0.45, green: 0.58, blue: 0.52, alpha: 1.0)
        ambientSky.light.intensity = 1500
        ambientSky.look(at: .zero, from: SIMD3<Float>(-3, 7, -4), relativeTo: nil)
        worldAnchor.addChild(ambientSky)
    }

    private func buildBaseGridData() {
        let ratio = pow(farRadius / innerRadius, 1.0 / Float(ringCount - 1))
        var radii: [Float] = [0]
        for k in 0..<ringCount { radii.append(innerRadius * pow(ratio, Float(k))) }

        for r in radii {
            for s in 0..<segmentCount {
                let theta = Float(s) / Float(segmentCount) * 2.0 * Float.pi
                let x = r * cos(theta)
                let z = r * sin(theta)
                basePositions.append(SIMD3<Float>(x, 0, z))
                baseRadii.append(r)
                let u = min(max(0.5 + x / foamExtent, 0.002), 0.998)
                let v = min(max(0.5 + z / foamExtent, 0.002), 0.998)
                baseUVs.append(SIMD2<Float>(u, v))
            }
        }

        for ring in 0..<ringCount {
            for s in 0..<segmentCount {
                let sn = (s + 1) % segmentCount
                let v00 = UInt32(ring * segmentCount + s)
                let v01 = UInt32(ring * segmentCount + sn)
                let v10 = UInt32((ring + 1) * segmentCount + s)
                let v11 = UInt32((ring + 1) * segmentCount + sn)
                triangleIndices.append(contentsOf: [v00, v01, v10, v01, v11, v10])
            }
        }
    }

    private func setupSwampWaterSurface() {
        standardWaterMaterial.baseColor = .init(tint: UIColor(red: 0.12, green: 0.22, blue: 0.18, alpha: 1.0))
        standardWaterMaterial.roughness = 0.14
        standardWaterMaterial.metallic = 0.08
        standardWaterMaterial.clearcoat = 0.35
        standardWaterMaterial.clearcoatRoughness = 0.16
        standardWaterMaterial.blending = .transparent(opacity: 0.88)
        standardWaterMaterial.faceCulling = .none

        let entity = ModelEntity(
            mesh: buildWaveMesh(time: 0.0, camX: 0, camZ: 0),
            materials: [standardWaterMaterial]
        )
        entity.position.y = lowestFloorY + currentWaterDepth
        worldAnchor.addChild(entity)
        self.waterEntity = entity

        let underMesh = MeshResource.generatePlane(
            width: murkDiameter,
            depth: murkDiameter,
            cornerRadius: murkDiameter / 2
        )
        var murkMat = PhysicallyBasedMaterial()
        murkMat.baseColor = .init(tint: UIColor(red: 0.05, green: 0.10, blue: 0.07, alpha: 1.0))
        murkMat.roughness = 0.9
        murkMat.blending = .transparent(opacity: 0.65)

        let underEntity = ModelEntity(mesh: underMesh, materials: [murkMat])
        underEntity.position.y = -0.08
        waterEntity?.addChild(underEntity)
        self.subsurfaceMurkEntity = underEntity
    }

    private func smoothStep(_ e0: Float, _ e1: Float, _ x: Float) -> Float {
        let t = min(max((x - e0) / (e1 - e0), 0), 1)
        return t * t * (3 - 2 * t)
    }

    private func buildWaveMesh(time t: Float, camX: Float, camZ: Float) -> MeshResource {
        var deformed = [SIMD3<Float>]()
        var normals = [SIMD3<Float>]()
        deformed.reserveCapacity(basePositions.count)
        normals.reserveCapacity(basePositions.count)

        for (i, p) in basePositions.enumerated() {
            let r = baseRadii[i]
            let worldX = camX + p.x
            let worldZ = camZ + p.z

            // Fade high-frequency ripples smoothly over distance
            let fade = 1.0 - smoothStep(12.0, 55.0, r)

            let p1 = worldX * 1.4 + worldZ * 0.8 + t * 1.9
            let p2 = worldX * -1.9 + worldZ * 1.5 - t * 2.3
            let p3 = worldX * 0.6 - worldZ * 2.5 + t * 2.9
            let p4 = (worldX - worldZ) * 3.4 - t * 3.8

            let height: Float = 0.035 * sin(p1) + 0.025 * sin(p2) + 0.015 * cos(p3) + 0.010 * sin(p4)
            let shiftX: Float = -0.015 * cos(p1) + 0.012 * cos(p2)
            let shiftZ: Float = -0.012 * cos(p1) - 0.014 * sin(p3)

            let y: Float = height * fade
            let dx: Float = shiftX * fade
            let dz: Float = shiftZ * fade

            // NOTE: The buggy manual mesh depression loop for `elevatedZones` has been entirely removed.
            // RealityKit's native invisible OcclusionMaterial volumes now perfectly handle hiding
            // the water inside curbs, planters, and false-positive walls without warping the water plane.

            deformed.append(SIMD3<Float>(p.x + dx, y, p.z + dz))

            let slopeX: Float = 0.035 * 1.4 * cos(p1) - 0.025 * 1.9 * cos(p2) - 0.015 * 0.6 * sin(p3) + 0.010 * 3.4 * cos(p4)
            let slopeZ: Float = 0.035 * 0.8 * cos(p1) + 0.025 * 1.5 * cos(p2) + 0.015 * 2.5 * sin(p3) - 0.010 * 3.4 * cos(p4)
            let dydx: Float = slopeX * fade
            let dydz: Float = slopeZ * fade
            normals.append(simd_normalize(SIMD3<Float>(-dydx * 2.5, 1.0, -dydz * 2.5)))
        }

        var descriptor = MeshDescriptor(name: "dynamicWaterMesh")
        descriptor.positions = MeshBuffer(deformed)
        descriptor.normals = MeshBuffer(normals)
        descriptor.textureCoordinates = MeshBuffer(baseUVs)
        descriptor.primitives = .triangles(triangleIndices)
        return (try? MeshResource.generate(from: [descriptor])) ?? MeshResource.generatePlane(width: 300, depth: 300)
    }

    private func startSimulationLoop() {
        updateSubscription = self.scene.subscribe(to: SceneEvents.Update.self) { [weak self] event in
            guard let self = self, let entity = self.waterEntity else { return }
            self.elapsedTime += Float(event.deltaTime)
            self.frameCounter += 1

            let camPos = self.cameraTransform.translation

            if !self.hasDetectedFloor {
                self.lowestFloorY = camPos.y - 1.5
            }

            let targetY = self.lowestFloorY + self.currentWaterDepth
            entity.position.y += (targetY - entity.position.y) * 0.15
            self.subsurfaceMurkEntity?.isEnabled = self.currentWaterDepth > 0.05

            if self.frameCounter % 2 == 0 {
                entity.model?.mesh = self.buildWaveMesh(
                    time: self.elapsedTime,
                    camX: camPos.x,
                    camZ: camPos.z
                )
                entity.position.x = camPos.x
                entity.position.z = camPos.z
            }

            self.lastShorelineUpdateTime += event.deltaTime
            if self.lastShorelineUpdateTime >= 0.3 {
                self.lastShorelineUpdateTime = 0
                if let newTex = self.generateCPUFoamTexture(centerX: camPos.x, centerZ: camPos.z, waterY: entity.position.y) {
                    self.standardWaterMaterial.baseColor = .init(tint: .white, texture: .init(newTex))
                    entity.model?.materials = [self.standardWaterMaterial]
                }
            }
        }
    }

    func updateWaterHeight(to height: Float) { self.currentWaterDepth = height }

    private func loadDepthModel() {
        depthQueue.async { [weak self] in
            var loaded: VNCoreMLModel?
            for name in ["DepthAnythingV2SmallF16", "DepthAnythingV2SmallF32"] {
                guard let url = Bundle.main.url(forResource: name, withExtension: "mlmodelc") else { continue }
                do {
                    let config = MLModelConfiguration()
                    config.computeUnits = .all
                    loaded = try VNCoreMLModel(for: MLModel(contentsOf: url, configuration: config))
                    break
                } catch {
                    print("FloodSight: failed to load \(name): \(error)")
                }
            }
            let result = loaded
            Task { @MainActor in
                self?.depthModel = result
            }
        }
    }

    private func currentImageOrientation() -> CGImagePropertyOrientation {
        switch window?.windowScene?.interfaceOrientation ?? .portrait {
        case .portraitUpsideDown: return .left
        case .landscapeLeft:      return .down
        case .landscapeRight:     return .up
        default:                  return .right
        }
    }

    private func collectLiDARSamples(from frame: ARFrame) -> [SIMD3<Float>] {
        guard let depth = frame.smoothedSceneDepth ?? frame.sceneDepth else { return [] }
        let map = depth.depthMap
        let conf = depth.confidenceMap
        let imgW = Float(frame.camera.imageResolution.width)
        let imgH = Float(frame.camera.imageResolution.height)

        CVPixelBufferLockBaseAddress(map, .readOnly)
        if let conf = conf { CVPixelBufferLockBaseAddress(conf, .readOnly) }
        defer {
            CVPixelBufferUnlockBaseAddress(map, .readOnly)
            if let conf = conf { CVPixelBufferUnlockBaseAddress(conf, .readOnly) }
        }
        guard let base = CVPixelBufferGetBaseAddress(map) else { return [] }

        let w = CVPixelBufferGetWidth(map)
        let h = CVPixelBufferGetHeight(map)
        let rowBytes = CVPixelBufferGetBytesPerRow(map)
        let confBase = conf.flatMap { CVPixelBufferGetBaseAddress($0) }
        let confRowBytes = conf.map { CVPixelBufferGetBytesPerRow($0) } ?? 0

        var out = [SIMD3<Float>]()
        for y in Swift.stride(from: 2, to: h, by: 6) {
            let row = base.advanced(by: y * rowBytes).assumingMemoryBound(to: Float32.self)
            for x in Swift.stride(from: 2, to: w, by: 6) {
                let z = row[x]
                guard z > 0.4, z < FarFieldDepth.lidarTrustRange else { continue }
                if let cb = confBase {
                    let c = cb.advanced(by: y * confRowBytes + x).assumingMemoryBound(to: UInt8.self).pointee
                    if c < UInt8(ARConfidenceLevel.medium.rawValue) { continue }
                }
                let u = (Float(x) + 0.5) / Float(w) * imgW
                let v = (Float(y) + 0.5) / Float(h) * imgH
                out.append(SIMD3<Float>(u, v, z))
            }
        }
        return out
    }

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        guard let model = depthModel,
              !isInferring,
              hasDetectedFloor,
              frame.timestamp - lastDepthTimestamp > depthInterval,
              case .normal = frame.camera.trackingState else { return }

        lastDepthTimestamp = frame.timestamp
        isInferring = true

        let cam = frame.camera
        let currentWaterY = waterEntity?.position.y ?? (lowestFloorY + currentWaterDepth)
        
        let ctx = DepthFrameContext(
            pixelBuffer: frame.capturedImage,
            camTransform: cam.transform,
            intrinsics: cam.intrinsics,
            imageSize: cam.imageResolution,
            orientation: currentImageOrientation(),
            floorY: lowestFloorY,
            waterY: currentWaterY,
            nearTrustRange: hasLiDAR ? FarFieldDepth.lidarTrustRange : 0,
            lidarSamples: collectLiDARSamples(from: frame)
        )

        depthQueue.async { [weak self] in
            let outcome = FarFieldDepth.run(model: model, ctx: ctx)
            Task { @MainActor in self?.finishDepthPass(outcome) }
        }
    }

    private func finishDepthPass(_ outcome: DepthOutcome) {
        isInferring = false
        switch outcome {
        case .occluder(let result):
            applyDepthOccluder(result)
        case .failed(let reason):
            if reason != lastDepthIssue {
                lastDepthIssue = reason
            }
        }
    }

    private func applyDepthOccluder(_ result: OccluderResult) {
        var descriptor = MeshDescriptor(name: "farFieldOccluder")
        descriptor.positions = MeshBuffer(result.positions)
        descriptor.primitives = .triangles(result.indices)
        guard let mesh = try? MeshResource.generate(from: [descriptor]) else { return }

        if let existing = depthOccluderEntity {
            existing.model?.mesh = mesh
        } else {
            let occluder = ModelEntity(mesh: mesh, materials: [OcclusionMaterial()])
            worldAnchor.addChild(occluder)
            depthOccluderEntity = occluder
        }
    }

    func session(_ session: ARSession, didAdd anchors: [ARAnchor]) { processAnchors(anchors) }
    func session(_ session: ARSession, didUpdate anchors: [ARAnchor]) { processAnchors(anchors) }

    func session(_ session: ARSession, didRemove anchors: [ARAnchor]) {
        for anchor in anchors {
            scannedMeshVertices.removeValue(forKey: anchor.identifier)
            elevatedZones.removeValue(forKey: anchor.identifier)
            if let occluder = planeOcclusionEntities.removeValue(forKey: anchor.identifier) {
                occluder.removeFromParent()
            }
        }
    }

    private func processAnchors(_ anchors: [ARAnchor]) {
        let camY = self.cameraTransform.translation.y

        for anchor in anchors {
            if let planeAnchor = anchor as? ARPlaneAnchor {
                let planeY = planeAnchor.transform.columns.3.y
                let width = max(0.2, planeAnchor.planeExtent.width)
                let height = max(0.2, planeAnchor.planeExtent.height)

                if planeAnchor.alignment == .horizontal {
                    let isReasonableFloor = planeY < (camY - 0.7)
                    
                    if isReasonableFloor {
                        if !hasDetectedFloor {
                            lowestFloorY = planeY
                            hasDetectedFloor = true
                        } else {
                            if planeAnchor.classification == .floor || planeY < lowestFloorY {
                                lowestFloorY = planeY
                            }
                        }
                    }

                    if hasDetectedFloor && planeY > lowestFloorY + 0.12 {
                        let boxHeight = max(0.15, planeY - lowestFloorY)
                        let mesh = MeshResource.generateBox(width: width, height: boxHeight, depth: height)
                        let occluder = planeOcclusionEntities[planeAnchor.identifier] ?? {
                            let e = ModelEntity(mesh: mesh, materials: [OcclusionMaterial()])
                            worldAnchor.addChild(e)
                            planeOcclusionEntities[planeAnchor.identifier] = e
                            return e
                        }()
                        occluder.model?.mesh = mesh

                        var transform = Transform(matrix: planeAnchor.transform)
                        let localOffset = SIMD3<Float>(planeAnchor.center.x, -boxHeight / 2.0, planeAnchor.center.z)
                        let worldPos = planeAnchor.transform * SIMD4<Float>(localOffset.x, localOffset.y, localOffset.z, 1.0)
                        transform.translation = SIMD3<Float>(worldPos.x, worldPos.y, worldPos.z)
                        occluder.transform = transform

                        elevatedZones[planeAnchor.identifier] = (
                            center: SIMD3<Float>(worldPos.x, planeY, worldPos.z),
                            radius: max(width, height) * 0.55,
                            topY: planeY
                        )
                    }
                } else if planeAnchor.alignment == .vertical {
                    let extendedHeight = max(height, abs(planeY - lowestFloorY) * 2.0 + 1.0)
                    let mesh = MeshResource.generateBox(width: width, height: 0.08, depth: extendedHeight)
                    let occluder = planeOcclusionEntities[planeAnchor.identifier] ?? {
                        let e = ModelEntity(mesh: mesh, materials: [OcclusionMaterial()])
                        worldAnchor.addChild(e)
                        planeOcclusionEntities[planeAnchor.identifier] = e
                        return e
                    }()
                    occluder.model?.mesh = mesh

                    var transform = Transform(matrix: planeAnchor.transform)
                    let worldPos = planeAnchor.transform * SIMD4<Float>(planeAnchor.center.x, 0, planeAnchor.center.z, 1.0)
                    transform.translation = SIMD3<Float>(worldPos.x, worldPos.y, worldPos.z)
                    occluder.transform = transform
                }
            }

            if let meshAnchor = anchor as? ARMeshAnchor {
                let geom = meshAnchor.geometry
                let vertices = geom.vertices
                let buffer = vertices.buffer.contents()
                var sampled: [SIMD3<Float>] = []
                for i in Swift.stride(from: 0, to: vertices.count, by: 8) {
                    let local = buffer.advanced(by: vertices.offset + i * vertices.stride).assumingMemoryBound(to: SIMD3<Float>.self).pointee
                    let world4 = meshAnchor.transform * SIMD4<Float>(local.x, local.y, local.z, 1.0)
                    sampled.append(SIMD3<Float>(world4.x, world4.y, world4.z))
                }
                scannedMeshVertices[meshAnchor.identifier] = sampled
            }
        }
    }

    private func generateCPUFoamTexture(centerX: Float, centerZ: Float, waterY: Float) -> TextureResource? {
        let texSize = 128
        var mask = [Float](repeating: 0.0, count: texSize * texSize)
        let halfExtent = foamExtent / 2.0

        for (_, points) in scannedMeshVertices {
            for pt in points {
                if pt.y - waterY >= -0.04 && pt.y - waterY <= 0.16 {
                    let u = (pt.x - centerX + halfExtent) / foamExtent
                    let v = (pt.z - centerZ + halfExtent) / foamExtent
                    if u > 0 && u < 1 && v > 0 && v < 1 {
                        stampFoam(into: &mask, size: texSize, cx: Int(u * Float(texSize)), cy: Int(v * Float(texSize)))
                    }
                }
            }
        }

        for (_, zone) in elevatedZones where zone.topY >= waterY - 0.05 {
            let u = (zone.center.x - centerX + halfExtent) / foamExtent
            let v = (zone.center.z - centerZ + halfExtent) / foamExtent
            if u > 0 && u < 1 && v > 0 && v < 1 {
                stampFoam(into: &mask, size: texSize, cx: Int(u * Float(texSize)), cy: Int(v * Float(texSize)))
            }
        }

        var pixels = [UInt8](repeating: 0, count: texSize * texSize * 4)
        for i in 0..<(texSize * texSize) {
            let foamIntensity = min(1.0, mask[i])
            var r = Float(30); var g = Float(55); var b = Float(45)

            if foamIntensity > 0 {
                r = r * (1.0 - foamIntensity) + 180.0 * foamIntensity
                g = g * (1.0 - foamIntensity) + 175.0 * foamIntensity
                b = b * (1.0 - foamIntensity) + 160.0 * foamIntensity
            }

            let idx = i * 4
            pixels[idx] = UInt8(clamping: Int(r))
            pixels[idx+1] = UInt8(clamping: Int(g))
            pixels[idx+2] = UInt8(clamping: Int(b))
            pixels[idx+3] = 255
        }

        guard let provider = CGDataProvider(data: Data(pixels) as CFData),
              let cgImage = CGImage(width: texSize, height: texSize, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: texSize * 4, space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue), provider: provider, decode: nil, shouldInterpolate: true, intent: .defaultIntent) else { return nil }
        return try? TextureResource.generate(from: cgImage, options: .init(semantic: .color))
    }

    private func stampFoam(into mask: inout [Float], size: Int, cx: Int, cy: Int) {
        let radius = 3; let r2 = Float(radius * radius)
        for dy in -radius...radius {
            let y = cy + dy; guard y >= 0 && y < size else { continue }
            for dx in -radius...radius {
                let x = cx + dx; guard x >= 0 && x < size else { continue }
                let dist2 = Float(dx * dx + dy * dy)
                if dist2 <= r2 { mask[y * size + x] += (1.0 - (dist2 / r2)) * 0.4 }
            }
        }
    }
}
