//
//  APIService.swift
//  FloodSight
//
//  Created by Matas Ambrukaitis on 10/24/26.
//

import Foundation
import CoreLocation
import MapKit
import Combine

// MARK: - Models
struct HealthResponse: Codable {
    let status: String
    let groundsourceAvailable: Bool
    enum CodingKeys: String, CodingKey {
        case status
        case groundsourceAvailable = "groundsource_available"
    }
}

// Reverted to match the updated POST /flood payload
struct FloodRequest: Codable {
    let latitude: Double
    let longitude: Double
    let bboxWgs84: [Double]
    let rainfallInches: Double?
    
    enum CodingKeys: String, CodingKey {
        case latitude
        case longitude
        case bboxWgs84 = "bbox_wgs84"
        case rainfallInches = "rainfall_inches"
    }
}

// New Models for POST /ar-depth
struct ARDepthRequest: Codable {
    let latitude: Double
    let longitude: Double
    let rainfallInches: Double?
    
    enum CodingKeys: String, CodingKey {
        case latitude
        case longitude
        case rainfallInches = "rainfall_inches"
    }
}

struct ARDepthResponse: Codable {
    let waterDepthMeters: Double
    let groundElevationMslMeters: Double
    let isFlooded: Bool
    let elevationDatum: String
    
    enum CodingKeys: String, CodingKey {
        case waterDepthMeters = "water_depth_meters"
        case groundElevationMslMeters = "ground_elevation_msl_meters"
        case isFlooded = "is_flooded"
        case elevationDatum = "elevation_datum"
    }
}

struct FloodResponse: Codable {
    let arFloodData: ARFloodData
    let mapContours: [FloodContour]
    enum CodingKeys: String, CodingKey {
        case arFloodData = "ar_flood_data"
        case mapContours = "map_contours"
    }
}

struct ARFloodData: Codable {
    let groundElevationMslMeters: Double
    let waterSurfaceMslMeters: Double
    let waterDepthMeters: Double
    let isFlooded: Bool
    let elevationDatum: String
    enum CodingKeys: String, CodingKey {
        case groundElevationMslMeters = "ground_elevation_msl_meters"
        case waterSurfaceMslMeters = "water_surface_msl_meters"
        case waterDepthMeters = "water_depth_meters"
        case isFlooded = "is_flooded"
        case elevationDatum = "elevation_datum"
    }
}

struct FloodContour: Codable, Identifiable {
    let id = UUID()
    let depthMeters: Double
    let coordinates: [ContourCoordinate]
    enum CodingKeys: String, CodingKey {
        case depthMeters = "depth_meters"
        case coordinates
    }
    var clCoordinates: [CLLocationCoordinate2D] {
        coordinates.map { CLLocationCoordinate2D(latitude: $0.lat, longitude: $0.lng) }
    }
}

struct ContourCoordinate: Codable {
    let lat: Double
    let lng: Double
}

struct WaterbodyProperties: Codable {
    let category: String?
    let name: String?
    let source: String?
    let sourceId: String?
    enum CodingKeys: String, CodingKey {
        case category, name, source
        case sourceId = "source_id"
    }
}

struct MappedWaterbody: Identifiable {
    let id = UUID()
    let name: String?
    let category: String?
    let coordinates: [CLLocationCoordinate2D]
}

struct APIErrorResponse: Codable {
    let error: String
}

enum FloodSightAPIError: LocalizedError {
    case invalidURL
    case serverError(statusCode: Int, message: String)
    case invalidResponse
    
    var errorDescription: String? {
        switch self {
        case .invalidURL: return "Invalid API URL."
        case .serverError(let code, let msg): return "Server Error (\(code)): \(msg)"
        case .invalidResponse: return "Invalid response from the FloodSight server."
        }
    }
}


// MARK: - API Service
class FloodSightAPIService {
    static let shared = FloodSightAPIService()
    var baseURL: String = "http://10.136.113.192:8000"
    private let session: URLSession

    private init() {
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 35.0
        config.timeoutIntervalForResource = 60.0
        self.session = URLSession(configuration: config)
    }

    func checkHealth() async throws -> HealthResponse {
        guard let url = URL(string: "\(baseURL)/health") else { throw FloodSightAPIError.invalidURL }
        let (data, response) = try await session.data(from: url)
        try validateResponse(data: data, response: response)
        return try JSONDecoder().decode(HealthResponse.self, from: data)
    }

    func fetchFloodData(requestBody: FloodRequest) async throws -> FloodResponse {
        guard let url = URL(string: "\(baseURL)/flood") else { throw FloodSightAPIError.invalidURL }
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONEncoder().encode(requestBody)

        let (data, response) = try await session.data(for: request)
        try validateResponse(data: data, response: response)
        return try JSONDecoder().decode(FloodResponse.self, from: data)
    }
    
    // New function for fast AR depth querying
    func fetchARDepth(requestBody: ARDepthRequest) async throws -> ARDepthResponse {
        guard let url = URL(string: "\(baseURL)/ar-depth") else { throw FloodSightAPIError.invalidURL }
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONEncoder().encode(requestBody)

        let (data, response) = try await session.data(for: request)
        try validateResponse(data: data, response: response)
        return try JSONDecoder().decode(ARDepthResponse.self, from: data)
    }

    func fetchWaterbodies(bboxWgs84: [Double]) async throws -> [MappedWaterbody] {
        guard bboxWgs84.count == 4 else { return [] }
        let bboxParam = bboxWgs84.map { String($0) }.joined(separator: ",")
        var components = URLComponents(string: "\(baseURL)/waterbodies")
        components?.queryItems = [URLQueryItem(name: "bbox_wgs84", value: bboxParam)]
        guard let url = components?.url else { throw FloodSightAPIError.invalidURL }

        let (data, response) = try await session.data(from: url)
        try validateResponse(data: data, response: response)

        let geoJSONObjects = try MKGeoJSONDecoder().decode(data)
        var results: [MappedWaterbody] = []

        for object in geoJSONObjects {
            if let feature = object as? MKGeoJSONFeature {
                var props: WaterbodyProperties?
                if let propData = feature.properties {
                    props = try? JSONDecoder().decode(WaterbodyProperties.self, from: propData)
                }
                for geom in feature.geometry {
                    if let polygon = geom as? MKPolygon {
                        results.append(MappedWaterbody(name: props?.name, category: props?.category, coordinates: polygon.coordinates))
                    } else if let multiPolygon = geom as? MKMultiPolygon {
                        for polygon in multiPolygon.polygons {
                            results.append(MappedWaterbody(name: props?.name, category: props?.category, coordinates: polygon.coordinates))
                        }
                    }
                }
            }
        }
        return results
    }

    private func validateResponse(data: Data, response: URLResponse) throws {
        guard let httpResponse = response as? HTTPURLResponse else { throw FloodSightAPIError.invalidResponse }
        guard (200...299).contains(httpResponse.statusCode) else {
            if let apiError = try? JSONDecoder().decode(APIErrorResponse.self, from: data) {
                throw FloodSightAPIError.serverError(statusCode: httpResponse.statusCode, message: apiError.error)
            }
            throw FloodSightAPIError.serverError(statusCode: httpResponse.statusCode, message: String(data: data, encoding: .utf8) ?? "Unknown error")
        }
    }
}

extension MKMultiPoint {
    var coordinates: [CLLocationCoordinate2D] {
        var coords = [CLLocationCoordinate2D](repeating: kCLLocationCoordinate2DInvalid, count: pointCount)
        getCoordinates(&coords, range: NSRange(location: 0, length: pointCount))
        return coords
    }
}

// MARK: - Shared ViewModel
@MainActor
class FloodViewModel: ObservableObject {
    @Published var rainfallInches: Double = 6.0
    @Published var arFloodData: ARFloodData?
    @Published var mapContours: [FloodContour] = []
    @Published var waterbodies: [MappedWaterbody] = []
    @Published var activeWaterDepthMeters: Float = 0.5
    @Published var isLoading: Bool = false
    @Published var errorMessage: String?
    @Published var serverHealthy: Bool = false

    // Called by the Map Tab
    func refreshAllData(visibleBbox: [Double]) async {
        guard visibleBbox.count == 4 else { return }
        
        isLoading = true
        errorMessage = nil

        // Calculate the center of the bounding box to satisfy the API's latitude/longitude requirement
        let centerLng = (visibleBbox[0] + visibleBbox[2]) / 2.0
        let centerLat = (visibleBbox[1] + visibleBbox[3]) / 2.0
        
        let floodReq = FloodRequest(
            latitude: centerLat,
            longitude: centerLng,
            bboxWgs84: visibleBbox,
            rainfallInches: rainfallInches
        )

        do {
            async let floodTask = FloodSightAPIService.shared.fetchFloodData(requestBody: floodReq)
            async let waterbodiesTask = FloodSightAPIService.shared.fetchWaterbodies(bboxWgs84: visibleBbox)

            let (floodRes, waterRes) = try await (floodTask, waterbodiesTask)

            self.arFloodData = floodRes.arFloodData
            self.mapContours = floodRes.mapContours.sorted { $0.depthMeters < $1.depthMeters }
            self.waterbodies = waterRes
            self.activeWaterDepthMeters = Float(floodRes.arFloodData.waterDepthMeters)
            self.serverHealthy = true
        } catch {
            self.errorMessage = error.localizedDescription
        }

        isLoading = false
    }
    
    // Called by the AR Tab when dragging the rainfall slider
    func updateARDepth(latitude: Double, longitude: Double) async {
        let req = ARDepthRequest(latitude: latitude, longitude: longitude, rainfallInches: rainfallInches)
        
        do {
            let res = try await FloodSightAPIService.shared.fetchARDepth(requestBody: req)
            self.activeWaterDepthMeters = Float(res.waterDepthMeters)
        } catch {
            print("Failed to fetch fast AR depth: \(error.localizedDescription)")
        }
    }
}
