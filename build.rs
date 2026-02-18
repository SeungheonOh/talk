fn main() {
    #[cfg(feature = "voxtral")]
    build_voxtral();
}

#[cfg(feature = "voxtral")]
fn build_voxtral() {
    use std::env;
    use std::process::Command;

    let vendor = "vendor/voxtral";
    let out_dir = env::var("OUT_DIR").unwrap();
    let is_macos = env::var("CARGO_CFG_TARGET_OS").unwrap() == "macos";
    let is_linux = env::var("CARGO_CFG_TARGET_OS").unwrap() == "linux";

    // Common C sources (no main.c, no mic capture)
    let c_sources = [
        "voxtral.c",
        "voxtral_kernels.c",
        "voxtral_audio.c",
        "voxtral_encoder.c",
        "voxtral_decoder.c",
        "voxtral_tokenizer.c",
        "voxtral_safetensors.c",
    ];

    let mut build = cc::Build::new();
    for src in &c_sources {
        build.file(format!("{}/{}", vendor, src));
    }
    build
        .include(vendor)
        .opt_level(3)
        .flag("-march=native")
        .flag("-ffast-math")
        .warnings(false);

    if is_macos {
        // Generate Metal shader source header via xxd
        let shader_header = format!("{}/voxtral_shaders_source.h", out_dir);
        let status = Command::new("xxd")
            .args(["-i", "voxtral_shaders.metal", &shader_header])
            .current_dir(vendor)
            .status()
            .expect("Failed to run xxd for Metal shader embedding");
        assert!(status.success(), "xxd failed");

        build
            .define("USE_BLAS", None)
            .define("ACCELERATE_NEW_LAPACK", None)
            .define("USE_METAL", None)
            .include(&out_dir)
            .file(format!("{}/voxtral_metal.m", vendor))
            .file(format!("{}/voxtral_metal_q8.m", vendor))
            .flag("-fobjc-arc");

        println!("cargo:rustc-link-lib=framework=Accelerate");
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=MetalPerformanceShaders");
        println!("cargo:rustc-link-lib=framework=MetalPerformanceShadersGraph");
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=AudioToolbox");
        println!("cargo:rustc-link-lib=framework=CoreFoundation");
    }

    if is_linux {
        build
            .define("USE_BLAS", None)
            .define("USE_OPENBLAS", None)
            .flag("-I/usr/include/openblas");

        println!("cargo:rustc-link-lib=openblas");
    }

    build.compile("voxtral");

    println!("cargo:rerun-if-changed={}", vendor);
}
