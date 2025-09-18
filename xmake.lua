add_rules("mode.debug", "mode.release")

-- 每次构建后自动更新 compile_commands.json
add_rules("plugin.compile_commands.autoupdate", {
    outputdir = ".vscode",  -- 指定输出目录
    lsp = "clangd"       -- 可选：生成后通知LSP服务器
})

add_repositories("local repo")
add_requires("cli11", "spdlog", "eigen", "openmp", "openblas", "parallel-hashmap", "mimalloc", "nlohmann_json")
add_requires("faiss-cpu")
-- add_requires("mkl", {system = true})

target("TDFANN")
    set_kind("binary")
    set_languages("c++20")
    add_links("stdc++")
    add_syslinks("pthread")

    add_packages("cli11", "spdlog", "eigen", "faiss-cpu", "openmp", "openblas", "parallel-hashmap", "mimalloc", "nlohmann_json")
    -- add_packages("mkl")
    -- add_ldflags("-lm", "-ldl")
    -- add_links("mkl_intel_lp64", "mkl_sequential", "mkl_core")
    add_files("src/main.cpp")
    add_defines("MI_MALLOC_OVERRIDE=1")  -- 让mimalloc覆盖全局malloc

    set_pcxxheader("include/PCH.hpp")  -- 设置预编译头文件
    add_cxxflags("-Wno-unknown-pragmas")      -- 忽略 #pragma system_header 警告
    add_cxxflags("-Wno-ignored-optimization-argument") -- 忽略 -Wno-gnu-line-marker 警告
    -- add_cxxflags("-H")  -- 输出头文件使用情况
    add_cxxflags("-Winvalid-pch") -- 检测PCH有效性
    
    -- if is_mode("release") then
    --     set_symbols("debug")   -- 在 Release 模式下也生成调试符号
    --     set_strip("none")      -- 明确禁止剥离符号
    --     -- set_optimize("fastest") -- 通常 Release 模式需要优化，保留这行
    -- end

    add_includedirs("include")

    set_warnings("all", "extra") -- 添加警告选项

    if is_mode("release") then
        set_policy("build.optimization.lto", true)
        add_cxflags("-O3 -march=native -mtune=native")
        -- set_strip("none")
    end

--
-- If you want to known more usage about xmake, please see https://xmake.io
--
-- ## FAQ
--
-- You can enter the project directory firstly before building project.
--
--   $ cd projectdir
--
-- 1. How to build project?
--
--   $ xmake
--
-- 2. How to configure project?
--
--   $ xmake f -p [macosx|linux|iphoneos ..] -a [x86_64|i386|arm64 ..] -m [debug|release]
--
-- 3. Where is the build output directory?
--
--   The default output directory is `./build` and you can configure the output directory.
--
--   $ xmake f -o outputdir
--   $ xmake
--
-- 4. How to run and debug target after building project?
--
--   $ xmake run [targetname]
--   $ xmake run -d [targetname]
--
-- 5. How to install target to the system directory or other output directory?
--
--   $ xmake install
--   $ xmake install -o installdir
--
-- 6. Add some frequently-used compilation flags in xmake.lua
--
-- @code
--    -- add debug and release modes
--    add_rules("mode.debug", "mode.release")
--
--    -- add macro definition
--    add_defines("NDEBUG", "_GNU_SOURCE=1")
--
--    -- set warning all as error
--    set_warnings("all", "error")
--
--    -- set language: c99, c++11
--    set_languages("c99", "c++11")
--
--    -- set optimization: none, faster, fastest, smallest
--    set_optimize("fastest")
--
--    -- add include search directories
--    add_includedirs("/usr/include", "/usr/local/include")
--
--    -- add link libraries and search directories
--    add_links("tbox")
--    add_linkdirs("/usr/local/lib", "/usr/lib")
--
--    -- add system link libraries
--    add_syslinks("z", "pthread")
--
--    -- add compilation and link flags
--    add_cxflags("-stdnolib", "-fno-strict-aliasing")
--    add_ldflags("-L/usr/local/lib", "-lpthread", {force = true})
--
-- @endcode
--

