//
//  VisualizerView.swift
//  FloodSight
//
//  Created by Matas Ambrukaitis on 10/23/26.
//

import SwiftUI
import RealityKit
import ARKit
 
struct VisualizerView: View {
   @State private var waterLevel: Float = 0.5 // Default 0.5 meters
   @State private var showControls: Bool = false
   
   var body: some View {
       ZStack {
           // AR View Container handles the camera and rendering
           ARWaterContainer(waterLevel: $waterLevel)
               .edgesIgnoringSafeArea(.all)
           
           // Minimal Toolbar Overlay
           VStack {
               HStack {
                   Spacer()
                   Button(action: {
                       showControls.toggle()
                   }) {
                       Image(systemName: "slider.vertical.3")
                           .font(.system(size: 24))
                           .padding()
                           .background(.ultraThinMaterial)
                           .clipShape(Circle())
                           .shadow(radius: 4)
                   }
                   .padding(.trailing, 20)
                   .padding(.top, 40)
               }
               Spacer()
           }
       }
       // Small Bottom Sheet for the Slider
       .sheet(isPresented: $showControls) {
           WaterControlSheet(waterLevel: $waterLevel)
               // presentationDetents keeps the sheet small and prevents it from going full screen
               .presentationDetents([.height(150)])
               .presentationDragIndicator(.visible)
       }
   }
}
 
// The UI for the Bottom Sheet
struct WaterControlSheet: View {
   @Binding var waterLevel: Float
   
   var body: some View {
       VStack(spacing: 20) {
           Text("Simulate Flood Level")
               .font(.headline)
           
           HStack {
               Text("0m")
               Slider(value: $waterLevel, in: -3...5, step: 0.1)
                   .tint(.blue)
               Text("5m")
           }
           .padding(.horizontal)
           
           Text(String(format: "%.1f Meters", waterLevel))
               .font(.subheadline)
               .foregroundColor(.secondary)
       }
       .padding()
   }
}

#Preview {
    VisualizerView()
}
