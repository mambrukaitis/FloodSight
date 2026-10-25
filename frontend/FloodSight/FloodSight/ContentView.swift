//
//  ContentView.swift
//  FloodSight
//
//  Created by Matas Ambrukaitis on 10/23/26.
//

import SwiftUI

struct ContentView: View {
    @StateObject private var floodVM = FloodViewModel()

        var body: some View {
            TabView {
                MapView(viewModel: floodVM)
                    .tabItem {
                        Label("Flood Map", systemImage: "map.fill")
                    }

                VisualizerView(viewModel: floodVM)
                    .tabItem {
                        Label("AR Visualizer", systemImage: "arkit")
                    }
            }
        }
    }

#Preview {
    ContentView()
}
