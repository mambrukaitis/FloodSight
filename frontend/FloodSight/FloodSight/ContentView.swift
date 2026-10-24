//
//  ContentView.swift
//  FloodSight
//
//  Created by Matas Ambrukaitis on 10/23/26.
//

import SwiftUI

struct ContentView: View {
    var body: some View {
        TabView {
            
            MapView()
                .tabItem {
                    Label("Map", systemImage: "map.fill")
                }
            
            VisualizerView()
                .tabItem {
                    Label("Visualizer", systemImage: "camera.viewfinder")
                }
        }
    }
}

#Preview {
    ContentView()
}
