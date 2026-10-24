//
//  MapView.swift
//  FloodSight
//
//  Created by Matas Ambrukaitis on 10/23/26.
//

import SwiftUI
import MapKit

struct MapView: View {
    @State private var cameraPosition: MapCameraPosition = .automatic
    
    var body: some View {
        Map(position: $cameraPosition) {
            
        }
        .mapStyle(.imagery(elevation: .realistic))
    }
}

#Preview {
    MapView()
}
