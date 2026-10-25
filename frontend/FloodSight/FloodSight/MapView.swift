//
//  MapView.swift
//  FloodSight
//
//  Created by Matas Ambrukaitis on 10/23/26.
//

import SwiftUI
import MapKit
import CoreLocation

struct MapView: View {
    @ObservedObject var viewModel: FloodViewModel
    @Namespace private var mapScope
    
    private let locationManager = CLLocationManager()
    
    @State private var cameraPosition: MapCameraPosition = .userLocation(fallback: .region(
        MKCoordinateRegion(
            center: CLLocationCoordinate2D(latitude: 29.6516, longitude: -82.3248),
            span: MKCoordinateSpan(latitudeDelta: 0.05, longitudeDelta: 0.05)
        )
    ))
    
    @State private var currentVisibleBbox: [Double] = [-82.3262, 29.6506, -82.3234, 29.6526]
    
    // State for Map Styles
    @State private var showMapStyleSheet = false
    @State private var selectedMapStyle: MapStyle = .imagery

    var body: some View {
        ZStack {
            Map(position: $cameraPosition, scope: mapScope) {
                UserAnnotation()

                ForEach(viewModel.waterbodies) { wb in
                    MapPolygon(coordinates: wb.coordinates)
                        .foregroundStyle(.teal.opacity(0.35))
                }

                ForEach(viewModel.mapContours) { contour in
                    MapPolygon(coordinates: contour.clCoordinates)
                        .foregroundStyle(colorForDepth(contour.depthMeters))
                }
            }
            .mapStyle(selectedMapStyle)
            .onMapCameraChange { context in
                let center = context.region.center
                let span = context.region.span
                
                let west = center.longitude - (span.longitudeDelta / 2)
                let south = center.latitude - (span.latitudeDelta / 2)
                let east = center.longitude + (span.longitudeDelta / 2)
                let north = center.latitude + (span.latitudeDelta / 2)
                
                currentVisibleBbox = [west, south, east, north]
            }
            // Keep only the scale view in automatic mapControls; we position Location & Compass manually
            .mapControls {
                MapScaleView()
            }
            .onAppear {
                locationManager.requestWhenInUseAuthorization()
            }

            // Floating Glass Overlays
            VStack {
                // Top Bar: Flood Depth Legend (Left) & Map Controls Stack (Right)
                HStack(alignment: .top) {
                    
                    // Top-Left Flood Depth Legend flush with top safe area
                    VStack(alignment: .leading, spacing: 6) {
                        Text("Flood Depth")
                            .font(.caption2.bold())
                        
                        Rectangle()
                            .fill(LinearGradient(
                                colors: [
                                    Color.cyan.opacity(0.5),
                                    Color.blue.opacity(0.6),
                                    Color.indigo.opacity(0.7),
                                    Color.purple.opacity(0.8)
                                ],
                                startPoint: .leading,
                                endPoint: .trailing
                            ))
                            .frame(height: 8)
                            .clipShape(Capsule())
                        
                        HStack {
                            Text("0.1m")
                            Spacer()
                            Text("1.0m+")
                        }
                        .font(.system(size: 9, weight: .medium))
                        .foregroundStyle(.secondary)
                    }
                    .padding(12)
                    .frame(width: 145)
                    .glassEffect(.regular, in: .rect(cornerRadius: 16))
                    
                    Spacer()
                    
                    // Top-Right Control Stack: Map Style -> Location Button -> Compass
                    VStack(alignment: .center, spacing: 10) {
                        Button {
                            showMapStyleSheet = true
                        } label: {
                            Image(systemName: "map")
                                .font(.system(size: 18, weight: .medium))
                                .foregroundStyle(.primary)
                                .frame(width: 44, height: 44)
                                .glassEffect(.regular.interactive(), in: .circle)
                        }
                        .buttonStyle(.plain)
                        
                        MapUserLocationButton(scope: mapScope)
                            .buttonBorderShape(.circle)
                        
                        MapCompass(scope: mapScope)
                    }
                }
                .padding(.horizontal, 16)
                .padding(.top, 6)
                
                Spacer()
                
                // Bottom Modeled Screening Scenario Glass Card
                VStack(spacing: 10) {
                    HStack {
                        VStack(alignment: .leading, spacing: 2) {
                            Text("Modeled Screening Scenario")
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                            
                            if let ar = viewModel.arFloodData {
                                Text(String(format: "Max Depth: %.2f m", ar.waterDepthMeters))
                                    .font(.subheadline.bold())
                            } else {
                                Text("Run simulation")
                                    .font(.subheadline.bold())
                            }
                        }
                        Spacer()
                        if viewModel.isLoading {
                            ProgressView()
                        } else {
                            Button("Simulate") {
                                Task { await viewModel.refreshAllData(visibleBbox: currentVisibleBbox) }
                            }
                            .buttonStyle(.glassProminent)
                            .controlSize(.small)
                        }
                    }

                    HStack {
                        Image(systemName: "cloud.heavyrain.fill")
                            .foregroundStyle(.blue)
                        Slider(value: $viewModel.rainfallInches, in: 0...20, step: 1)
                        Text("\(Int(viewModel.rainfallInches)) in")
                            .font(.caption.monospacedDigit())
                            .frame(width: 40, alignment: .trailing)
                    }

                    if let error = viewModel.errorMessage {
                        Text(error)
                            .font(.caption2)
                            .foregroundStyle(.red)
                            .lineLimit(2)
                    }
                }
                .padding(16)
                .glassEffect(.regular, in: .rect(cornerRadius: 22))
                .padding()
            }
        }
        .mapScope(mapScope)
        .task {
            if viewModel.arFloodData == nil {
                await viewModel.refreshAllData(visibleBbox: currentVisibleBbox)
            }
        }
        .sheet(isPresented: $showMapStyleSheet) {
            MapStyleSelectorView(selectedStyle: $selectedMapStyle)
                .presentationDetents([.height(220)])
                .presentationDragIndicator(.visible)
        }
    }

    private func colorForDepth(_ depth: Double) -> Color {
        switch depth {
        case ..<0.15: return Color.cyan.opacity(0.40)
        case ..<0.25: return Color.blue.opacity(0.45)
        case ..<0.50: return Color.indigo.opacity(0.55)
        default:      return Color.purple.opacity(0.65)
        }
    }
}

// MARK: - Liquid Glass Map Style Sheet
struct MapStyleSelectorView: View {
    @Binding var selectedStyle: MapStyle
    @Environment(\.dismiss) var dismiss
    
    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            Text("Select Map Type")
                .font(.headline)
                .padding(.top, 20)
            
            HStack(spacing: 16) {
                StyleButton(title: "Standard", icon: "map", isSelected: isStandard) {
                    selectedStyle = .standard
                    dismiss()
                }
                
                StyleButton(title: "Satellite", icon: "globe.americas.fill", isSelected: isSatellite) {
                    selectedStyle = .imagery
                    dismiss()
                }
                
                StyleButton(title: "Hybrid", icon: "map.circle", isSelected: isHybrid) {
                    selectedStyle = .hybrid
                    dismiss()
                }
            }
            Spacer()
        }
        .padding(.horizontal, 20)
    }
    
    var isStandard: Bool { String(describing: selectedStyle) == String(describing: MapStyle.standard) }
    var isSatellite: Bool { String(describing: selectedStyle) == String(describing: MapStyle.imagery) }
    var isHybrid: Bool { String(describing: selectedStyle) == String(describing: MapStyle.hybrid) }
}

struct StyleButton: View {
    let title: String
    let icon: String
    let isSelected: Bool
    let action: () -> Void
    
    var body: some View {
        Button(action: action) {
            VStack(spacing: 8) {
                Image(systemName: icon)
                    .font(.system(size: 28, weight: .medium))
                    .foregroundStyle(isSelected ? .white : .primary)
                    .frame(maxWidth: .infinity)
                    .frame(height: 68)
                    .glassEffect(
                        isSelected ? .regular.tint(.blue).interactive() : .regular.interactive(),
                        in: .rect(cornerRadius: 16)
                    )
                
                Text(title)
                    .font(.caption)
                    .fontWeight(isSelected ? .semibold : .regular)
                    .foregroundStyle(.primary)
            }
        }
        .buttonStyle(.plain)
    }
}
