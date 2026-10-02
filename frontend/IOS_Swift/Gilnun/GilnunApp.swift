//
//  GilnunApp.swift
//  Gilnun
//
//  Created by JoMinHui on 4/10/26.
//

import SwiftUI
import UIKit

@main
struct GilnunApp: App {
    @Environment(\.scenePhase) private var scenePhase

    var body: some Scene {
        WindowGroup {
            ContentView()
        }
        .onChange(of: scenePhase, initial: true) { _, phase in
            // Auto-lock would stop the camera mid-walk, so keep the screen on while the app is in use.
            UIApplication.shared.isIdleTimerDisabled = phase == .active
        }
    }
}
