/// Hide the app from the Dock and Cmd-Tab switcher.
#[cfg(target_os = "macos")]
pub fn hide_from_dock() {
    use objc2_app_kit::{NSApplication, NSApplicationActivationPolicy};
    use objc2_foundation::MainThreadMarker;

    let mtm = MainThreadMarker::new()
        .expect("hide_from_dock must be called on the main thread");
    let app = NSApplication::sharedApplication(mtm);
    app.setActivationPolicy(NSApplicationActivationPolicy::Accessory);
}

/// After iced/winit creates the NSWindow, convert it to a non-activating
/// NSPanel and configure it for overlay behaviour (transparent, all spaces,
/// click-through). Also sets up the menu bar tray icon.
/// Dispatched to the main thread via GCD after a short delay
/// so iced has time to create the window.
#[cfg(target_os = "macos")]
pub fn configure_as_overlay_panel() {
    use std::os::raw::c_void;

    #[link(name = "System", kind = "dylib")]
    unsafe extern "C" {
        static _dispatch_main_q: u8;
        fn dispatch_async_f(
            queue: *const c_void,
            context: *mut c_void,
            work: unsafe extern "C" fn(*mut c_void),
        );
    }

    unsafe extern "C" fn do_configure(_ctx: *mut c_void) {
        use objc2::runtime::AnyObject;
        use objc2::{ClassType, MainThreadMarker};
        use objc2_app_kit::{
            NSApplication, NSApplicationActivationPolicy, NSColor, NSPanel,
            NSWindowCollectionBehavior, NSWindowStyleMask,
        };

        unsafe {
            let mtm = MainThreadMarker::new().expect("must be on main thread");
            let app = NSApplication::sharedApplication(mtm);
            let windows = app.windows();

            for i in 0..windows.count() {
                let win = windows.objectAtIndex(i);

                AnyObject::set_class(
                    &*((&*win) as *const _ as *const AnyObject),
                    NSPanel::class(),
                );
                let panel: &NSPanel = &*(&*win as *const _ as *const NSPanel);

                let mask = panel.styleMask();
                panel.setStyleMask(mask | NSWindowStyleMask::NonactivatingPanel);
                panel.setFloatingPanel(true);
                panel.setBecomesKeyOnlyIfNeeded(true);

                panel.setCollectionBehavior(
                    NSWindowCollectionBehavior::CanJoinAllSpaces
                        | NSWindowCollectionBehavior::Stationary
                        | NSWindowCollectionBehavior::IgnoresCycle
                        | NSWindowCollectionBehavior::FullScreenAuxiliary,
                );

                panel.setHasShadow(false);

                let clear = NSColor::clearColor();
                panel.setBackgroundColor(Some(&clear));
                panel.setOpaque(false);

                panel.setIgnoresMouseEvents(true);
            }

            // Re-apply Accessory policy — iced/winit may have reset it
            app.setActivationPolicy(NSApplicationActivationPolicy::Accessory);
            app.deactivate();
        }

        setup_tray_icon_inner();
    }

    std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_millis(200));
        unsafe {
            let main_q = &raw const _dispatch_main_q as *const c_void;
            dispatch_async_f(main_q, std::ptr::null_mut(), do_configure);
        }
    });
}

// ---------------------------------------------------------------------------
// Tray icon with state indicator
// ---------------------------------------------------------------------------

#[cfg(target_os = "macos")]
mod tray {
    use std::sync::atomic::{AtomicPtr, Ordering};
    use std::os::raw::c_void;

    /// Raw pointer to the NSStatusBarButton so we can update its title.
    static BUTTON_PTR: AtomicPtr<c_void> = AtomicPtr::new(std::ptr::null_mut());

    const ICON_STANDBY: &str = "○ ";
    const ICON_CAPTURING: &str = "● ";

    pub fn store_button(ptr: *mut c_void) {
        BUTTON_PTR.store(ptr, Ordering::Release);
    }

    pub fn set_capturing(active: bool) {
        use objc2_app_kit::NSStatusBarButton;
        use objc2_foundation::NSString;

        let ptr = BUTTON_PTR.load(Ordering::Acquire);
        if ptr.is_null() {
            return;
        }
        let icon = if active { ICON_CAPTURING } else { ICON_STANDBY };
        unsafe {
            let button: &NSStatusBarButton = &*(ptr as *const NSStatusBarButton);
            button.setTitle(&NSString::from_str(icon));
        }
    }
}

/// Update the tray icon to reflect capturing state.
#[cfg(target_os = "macos")]
pub fn set_tray_capturing(active: bool) {
    tray::set_capturing(active);
}

#[cfg(not(target_os = "macos"))]
pub fn set_tray_capturing(_active: bool) {}

/// Create a menu bar status item with a Quit option.
/// Called from the GCD dispatch block after the run loop is active.
#[cfg(target_os = "macos")]
fn setup_tray_icon_inner() {
    use objc2::{sel, MainThreadOnly};
    use objc2_app_kit::{NSApplication, NSMenu, NSMenuItem, NSStatusBar};
    use objc2_core_foundation::CGFloat;
    use objc2_foundation::{MainThreadMarker, NSString};

    let mtm = MainThreadMarker::new().expect("must be on main thread");

    let status_bar = NSStatusBar::systemStatusBar();
    let status_item = status_bar.statusItemWithLength(CGFloat::from(-1.0_f32));

    if let Some(button) = status_item.button(mtm) {
        button.setTitle(&NSString::from_str("○ "));
        // Store raw pointer for later updates
        let ptr = &*button as *const _ as *mut std::os::raw::c_void;
        tray::store_button(ptr);
    }

    let menu = NSMenu::initWithTitle(NSMenu::alloc(mtm), &NSString::from_str("Voicer"));

    let quit_title = NSString::from_str("Quit Voicer");
    let quit_key = NSString::from_str("q");
    let quit_item = unsafe {
        NSMenuItem::initWithTitle_action_keyEquivalent(
            NSMenuItem::alloc(mtm),
            &quit_title,
            Some(sel!(terminate:)),
            &quit_key,
        )
    };
    let app = NSApplication::sharedApplication(mtm);
    unsafe { quit_item.setTarget(Some(&app)) };
    menu.addItem(&quit_item);

    status_item.setMenu(Some(&menu));

    std::mem::forget(quit_item);
    std::mem::forget(menu);
    std::mem::forget(status_item);
}

/// Get the main display dimensions (width, height).
#[cfg(target_os = "macos")]
pub fn main_display_size() -> (f32, f32) {
    #[repr(C)]
    #[derive(Default)]
    struct CGRect {
        x: f64,
        y: f64,
        w: f64,
        h: f64,
    }

    #[link(name = "CoreGraphics", kind = "framework")]
    unsafe extern "C" {
        fn CGMainDisplayID() -> u32;
        fn CGDisplayBounds(display: u32) -> CGRect;
    }

    unsafe {
        let display = CGMainDisplayID();
        let bounds = CGDisplayBounds(display);
        (bounds.w as f32, bounds.h as f32)
    }
}

/// Sample average brightness of a screen region. Returns 0.0 (black) to 1.0 (white).
#[cfg(target_os = "macos")]
pub fn sample_background_brightness(x: f32, y: f32, w: f32, h: f32) -> f32 {
    use std::os::raw::c_void;

    #[repr(C)]
    struct CGRect {
        x: f64,
        y: f64,
        w: f64,
        h: f64,
    }

    #[link(name = "CoreGraphics", kind = "framework")]
    unsafe extern "C" {
        fn CGMainDisplayID() -> u32;
        fn CGDisplayCreateImageForRect(display: u32, rect: CGRect) -> *const c_void;
        fn CGImageGetWidth(image: *const c_void) -> usize;
        fn CGImageGetHeight(image: *const c_void) -> usize;
        fn CGImageGetBytesPerRow(image: *const c_void) -> usize;
        fn CGImageGetBitsPerPixel(image: *const c_void) -> usize;
        fn CGImageGetDataProvider(image: *const c_void) -> *const c_void;
        fn CGDataProviderCopyData(provider: *const c_void) -> *const c_void;
    }

    #[link(name = "CoreFoundation", kind = "framework")]
    unsafe extern "C" {
        fn CFDataGetBytePtr(data: *const c_void) -> *const u8;
        fn CFDataGetLength(data: *const c_void) -> isize;
        fn CFRelease(cf: *const c_void);
    }

    unsafe {
        let display = CGMainDisplayID();
        let rect = CGRect { x: x as f64, y: y as f64, w: w as f64, h: h as f64 };
        let image = CGDisplayCreateImageForRect(display, rect);
        if image.is_null() {
            return 0.5;
        }

        let provider = CGImageGetDataProvider(image);
        let data = CGDataProviderCopyData(provider);
        if data.is_null() {
            CFRelease(image);
            return 0.5;
        }

        let ptr = CFDataGetBytePtr(data);
        let len = CFDataGetLength(data) as usize;
        let width = CGImageGetWidth(image);
        let height = CGImageGetHeight(image);
        let bytes_per_row = CGImageGetBytesPerRow(image);
        let bpp = CGImageGetBitsPerPixel(image) / 8;

        let mut total = 0.0f64;
        let mut count = 0u64;

        for row in (0..height).step_by(4) {
            for col in (0..width).step_by(4) {
                let offset = row * bytes_per_row + col * bpp;
                if offset + 2 < len {
                    let c0 = *ptr.add(offset) as f64 / 255.0;
                    let c1 = *ptr.add(offset + 1) as f64 / 255.0;
                    let c2 = *ptr.add(offset + 2) as f64 / 255.0;
                    // Green channel dominates luminance regardless of BGRA/RGBA order
                    total += 0.299 * c0 + 0.587 * c1 + 0.114 * c2;
                    count += 1;
                }
            }
        }

        CFRelease(data);
        CFRelease(image);

        if count == 0 { 0.5 } else { (total / count as f64) as f32 }
    }
}

#[cfg(not(target_os = "macos"))]
pub fn sample_background_brightness(_x: f32, _y: f32, _w: f32, _h: f32) -> f32 {
    0.5
}

#[cfg(not(target_os = "macos"))]
pub fn hide_from_dock() {}

#[cfg(not(target_os = "macos"))]
pub fn configure_as_overlay_panel() {}

#[cfg(not(target_os = "macos"))]
pub fn main_display_size() -> (f32, f32) {
    (1920.0, 1080.0)
}
