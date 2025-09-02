package("faiss-cpu")
set_homepage("https://github.com/facebookresearch/faiss")
set_description("Faiss CPU-only")
set_urls("https://github.com/facebookresearch/faiss/archive/refs/tags/v$(version).tar.gz")
add_versions("1.12.0", "561376d1a44771bf1230fabeef9c81643468009b45a585382cf38d3a7a94990a")
add_deps("cmake")
on_install(function (package)
    local configs = {
        "-DFAISS_ENABLE_GPU=OFF",
        "-DFAISS_ENABLE_C_API=ON",
        "-DFAISS_ENABLE_PYTHON=OFF",
        -- "-DFAISS_ENABLE_MPI=OFF",
        "-DBUILD_TESTING=OFF",
        "-DCMAKE_DISABLE_FIND_PACKAGE_CUDA=ON",
        "-DCUDA_TOOLKIT_ROOT_DIR=",
        "-DCMAKE_CUDA_COMPILER=",
    }
    import("package.tools.cmake").install(package, configs)
end)

on_test(function (package)
    assert(package:has_cxxtypes("faiss::IndexFlatL2", {includes = "faiss/IndexFlat.h"}))
end)