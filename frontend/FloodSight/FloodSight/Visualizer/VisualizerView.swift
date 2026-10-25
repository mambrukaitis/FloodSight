//
//  VisualizerView.swift
//  FloodSight
//
//  Created by Matas Ambrukaitis on 10/23/26.
//

import SwiftUI
import RealityKit

struct VisualizerView: View {
    @ObservedObject var viewModel: FloodViewModel
    
    @State private var isManualMode: Bool = false

    var body: some View {
        NavigationStack {
            ZStack(alignment: .topTrailing) {
                
                // 1. Live AR Camera View
                ARWaterContainer(waterLevel: $viewModel.activeWaterDepthMeters)
                    .ignoresSafeArea()
                
                // 2. Slim Liquid Glass Vertical Slider right under the toolbar button
                VStack(spacing: 10) {
                    // Top Mark
                    Text(isManualMode ? "3m" : "20\"")
                        .font(.system(size: 11, weight: .bold, design: .rounded))
                        .foregroundStyle(.primary)
                    
                    // Rotated Slider constrained to a slim 28pt layout width
                    Group {
                        if isManualMode {
                            Slider(
                                value: $viewModel.activeWaterDepthMeters,
                                in: 0...3
                            )
                        } else {
                            Slider(
                                value: $viewModel.rainfallInches,
                                in: 0...20
                            ) { editing in
                                if !editing {
                                    Task {
                                        await viewModel.updateARDepth(latitude: 29.6516, longitude: -82.3248)
                                    }
                                }
                            }
                        }
                    }
                    .tint(.cyan)
                    .frame(width: 160)
                    .rotationEffect(.degrees(-90))
                    .frame(width: 28, height: 160)
                    
                    // Bottom Mark
                    Text(isManualMode ? "0m" : "0\"")
                        .font(.system(size: 11, weight: .bold, design: .rounded))
                        .foregroundStyle(.primary)
                }
                .padding(.vertical, 14)
                .padding(.horizontal, 8) // 28pt + 16pt padding = 44pt total width
                .glassEffect(.regular, in: .capsule)
                .padding(.top, 8)
                .padding(.trailing, 16)
            }
            // 3. Native Liquid Glass Toolbar Menu (without the inner circle cutout)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Menu {
                        Toggle(isOn: $isManualMode) {
                            Label("Manual Water Level", systemImage: "hand.point.up.left")
                        }
                    } label: {
                        Label("Options", systemImage: "ellipsis")
                    }
                }
            }
            .toolbarBackgroundVisibility(.hidden, for: .navigationBar)
        }
        .onAppear {
            if !viewModel.serverHealthy {
                isManualMode = true
            }
        }
        .onChange(of: isManualMode) { _, newValue in
            if !newValue {
                Task {
                    await viewModel.updateARDepth(latitude: 29.6516, longitude: -82.3248)
                }
            }
        }
    }
}
