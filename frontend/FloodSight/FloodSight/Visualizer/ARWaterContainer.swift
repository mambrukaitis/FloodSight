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

struct ARWaterContainer: UIViewRepresentable {
    @Binding var waterLevel: Float
    func makeUIView(context: Context) -> CustomARView { return CustomARView(frame: .zero) }
    func updateUIView(_ uiView: CustomARView, context: Context) { uiView.updateWaterHeight(to: waterLevel) }
}

class CustomARView: ARView, ARSessionDelegate {
    
    private var worldAnchor: AnchorEntity!
    private var waterEntity: ModelEntity?
    private var subsurfaceMurkEntity: ModelEntity?
    private var standardWaterMaterial = PhysicallyBasedMaterial()
    private var updateSubscription: Cancellable?
    
    private var lowestFloorY: Float = -1.3
    private var hasDetectedFloor: Bool = false
    private var currentWaterDepth: Float = 0.5
    
    // CPU-Optimized 45x45 grid for 60fps 3D waves
    private let gridResolution: Int = 45
    private let meshSize: Float = 24.0
    private var basePositions: [SIMD3<Float>] = []
    private var baseUVs: [SIMD2<Float>] = []
    private var triangleIndices: [UInt32] = []
    
    private var elapsedTime: Float = 0.0
    private var frameCounter: Int = 0
    
    private var scannedMeshVertices: [UUID: [SIMD3<Float>]] = [:]
    private var verticalPlaneBounds: [UUID: (transform: simd_float4x4, extent: SIMD3<Float>)] = [:]
    private var lastShorelineUpdateTime: TimeInterval = 0
    
    required init(frame frameRect: CGRect) {
        super.init(frame: frameRect)
        setupARAndStableOcclusion()
        setupLighting()
        buildBaseGridData()
        setupSwampWaterSurface()
        startSimulationLoop()
    }
    
    required init?(coder decoder: NSCoder) {
        super.init(coder: decoder)
        setupARAndStableOcclusion()
        setupLighting()
        buildBaseGridData()
        setupSwampWaterSurface()
        startSimulationLoop()
    }
    
    // MARK: - 1. Stable Occlusion (Fixes the ForceClear Crash)
    private func setupARAndStableOcclusion() {
        self.session.delegate = self
        let config = ARWorldTrackingConfiguration()
        config.planeDetection = [.horizontal, .vertical]
        config.environmentTexturing = .automatic
        
        if ARWorldTrackingConfiguration.supportsSceneReconstruction(.mesh) {
            config.sceneReconstruction = .mesh
        }
        
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.smoothedSceneDepth) {
            config.frameSemantics.insert(.smoothedSceneDepth)
        }
        
        // ONLY .occlusion. We strictly exclude .receivesLighting to prevent LiDAR shadow crashes.
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
        let step = meshSize / Float(gridResolution)
        let halfSize = meshSize / 2.0
        for z in 0...gridResolution {
            for x in 0...gridResolution {
                basePositions.append(SIMD3<Float>(Float(x) * step - halfSize, 0, Float(z) * step - halfSize))
                baseUVs.append(SIMD2<Float>(Float(x) / Float(gridResolution), Float(z) / Float(gridResolution)))
            }
        }
        let rowLen = UInt32(gridResolution + 1)
        for z in 0..<gridResolution {
            for x in 0..<gridResolution {
                let tl = UInt32(z) * rowLen + UInt32(x)
                let tr = tl + 1; let bl = (UInt32(z) + 1) * rowLen + UInt32(x); let br = bl + 1
                triangleIndices.append(contentsOf: [tl, bl, tr, tr, bl, br])
            }
        }
    }
    
    // MARK: - 2. Native PBR Material (Fixes the Index 28 Crash)
    private func setupSwampWaterSurface() {
        // Deep murky green-brown base
        standardWaterMaterial.baseColor = .init(tint: UIColor(red: 0.12, green: 0.22, blue: 0.18, alpha: 1.0))
        standardWaterMaterial.roughness = 0.05
        standardWaterMaterial.metallic = 0.1
        standardWaterMaterial.clearcoat = 1.0
        standardWaterMaterial.clearcoatRoughness = 0.05
        standardWaterMaterial.blending = .transparent(opacity: 0.85) // Native transparency is safe
        standardWaterMaterial.faceCulling = .none
        
        let entity = ModelEntity(mesh: buildWaveMesh(time: 0.0), materials: [standardWaterMaterial])
        entity.position.y = lowestFloorY + currentWaterDepth
        worldAnchor.addChild(entity)
        self.waterEntity = entity
        
        let underMesh = MeshResource.generatePlane(width: meshSize, depth: meshSize)
        var murkMat = PhysicallyBasedMaterial()
        murkMat.baseColor = .init(tint: UIColor(red: 0.05, green: 0.09, blue: 0.06, alpha: 1.0))
        murkMat.blending = .transparent(opacity: 0.6)
        
        let underEntity = ModelEntity(mesh: underMesh, materials: [murkMat])
        underEntity.position.y = -0.15
        waterEntity?.addChild(underEntity)
        self.subsurfaceMurkEntity = underEntity
    }
    
    private func buildWaveMesh(time t: Float) -> MeshResource {
        var deformed = [SIMD3<Float>]()
        var normals = [SIMD3<Float>]()
        deformed.reserveCapacity(basePositions.count)
        normals.reserveCapacity(basePositions.count)
        
        for p in basePositions {
            let p1 = p.x * 1.4 + p.z * 0.8 + t * 1.9
            let p2 = p.x * -1.9 + p.z * 1.5 - t * 2.3
            let p3 = p.x * 0.6 - p.z * 2.5 + t * 2.9
            let p4 = (p.x - p.z) * 3.4 - t * 3.8
            
            let y = 0.035 * sin(p1) + 0.025 * sin(p2) + 0.015 * cos(p3) + 0.010 * sin(p4)
            let dx = -0.015 * cos(p1) + 0.012 * cos(p2)
            let dz = -0.012 * cos(p1) - 0.014 * sin(p3)
            
            deformed.append(SIMD3<Float>(p.x + dx, y, p.z + dz))
            
            let dydx = 0.035 * 1.4 * cos(p1) - 0.025 * 1.9 * cos(p2) - 0.015 * 0.6 * sin(p3) + 0.010 * 3.4 * cos(p4)
            let dydz = 0.035 * 0.8 * cos(p1) + 0.025 * 1.5 * cos(p2) + 0.015 * 2.5 * sin(p3) - 0.010 * 3.4 * cos(p4)
            normals.append(simd_normalize(SIMD3<Float>(-dydx * 2.5, 1.0, -dydz * 2.5)))
        }
        
        var descriptor = MeshDescriptor(name: "dynamicWaterMesh")
        descriptor.positions = MeshBuffer(deformed)
        descriptor.normals = MeshBuffer(normals)
        descriptor.textureCoordinates = MeshBuffer(baseUVs)
        descriptor.primitives = .triangles(triangleIndices)
        return (try? MeshResource.generate(from: [descriptor])) ?? MeshResource.generatePlane(width: meshSize, depth: meshSize)
    }
    
    private func startSimulationLoop() {
        updateSubscription = self.scene.subscribe(to: SceneEvents.Update.self) { [weak self] event in
            guard let self = self, let entity = self.waterEntity else { return }
            self.elapsedTime += Float(event.deltaTime)
            self.frameCounter += 1
            
            let camPos = self.cameraTransform.translation
            if self.frameCounter % 2 == 0 {
                entity.model?.mesh = self.buildWaveMesh(time: self.elapsedTime)
                entity.position.x = camPos.x
                entity.position.z = camPos.z
            }
            
            entity.position.y += ((self.lowestFloorY + self.currentWaterDepth) - entity.position.y) * 0.15
            self.subsurfaceMurkEntity?.isEnabled = self.currentWaterDepth > 0.08
            
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
    
    // MARK: - 3. LiDAR Extraction
    func session(_ session: ARSession, didAdd anchors: [ARAnchor]) { processAnchors(anchors) }
    func session(_ session: ARSession, didUpdate anchors: [ARAnchor]) { processAnchors(anchors) }
    
    private func processAnchors(_ anchors: [ARAnchor]) {
        for anchor in anchors {
            if let planeAnchor = anchor as? ARPlaneAnchor {
                if planeAnchor.alignment == .horizontal {
                    let y = planeAnchor.transform.columns.3.y
                    if !hasDetectedFloor || (planeAnchor.classification == .floor && y < lowestFloorY + 0.15) || y < lowestFloorY {
                        lowestFloorY = y; hasDetectedFloor = true
                    }
                } else if planeAnchor.alignment == .vertical {
                    verticalPlaneBounds[planeAnchor.identifier] = (planeAnchor.transform, SIMD3<Float>(planeAnchor.planeExtent.width, 0, planeAnchor.planeExtent.height))
                }
            }
            if let meshAnchor = anchor as? ARMeshAnchor {
                let geom = meshAnchor.geometry; let vertices = geom.vertices; let buffer = vertices.buffer.contents()
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
    
    // MARK: - 4. CPU Procedural Foam & Swamp Color Texture
    private func generateCPUFoamTexture(centerX: Float, centerZ: Float, waterY: Float) -> TextureResource? {
        let texSize = 128
        var mask = [Float](repeating: 0.0, count: texSize * texSize)
        let halfMesh = meshSize / 2.0
        
        for (_, points) in scannedMeshVertices {
            for pt in points {
                if pt.y - waterY >= -0.04 && pt.y - waterY <= 0.16 {
                    let u = (pt.x - centerX + halfMesh) / meshSize
                    let v = (pt.z - centerZ + halfMesh) / meshSize
                    if u > 0 && u < 1 && v > 0 && v < 1 {
                        stampFoam(into: &mask, size: texSize, cx: Int(u * Float(texSize)), cy: Int(v * Float(texSize)))
                    }
                }
            }
        }
        
        var pixels = [UInt8](repeating: 0, count: texSize * texSize * 4)
        for i in 0..<(texSize * texSize) {
            let foamIntensity = min(1.0, mask[i])
            // Swamp Water Base: Dark Olive Green
            var r = Float(30); var g = Float(55); var b = Float(45)
            
            // Blend to Dirty Foam color (Khaki White) based on shoreline proximity
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
