function(get_arch_flags architecture compiler)

    # Westmere
    if("${architecture}" STREQUAL "wsm")
        set(NO_REDZONE ON PARENT_SCOPE)
        set(CPU_ARCH_FLAGS "-march=westmere" PARENT_SCOPE)
    
    # Sandy Bridge
    elseif("${architecture}" STREQUAL "snb")
        set(NO_REDZONE ON PARENT_SCOPE)
        set(CPU_ARCH_FLAGS "-march=sandybridge" PARENT_SCOPE)

    # Haswell
    elseif("${architecture}" STREQUAL "hsw")
        set(NO_REDZONE ON PARENT_SCOPE)
        set(CPU_ARCH_FLAGS "-march=haswell" PARENT_SCOPE)

    # Knights Landing
    elseif("${architecture}" STREQUAL "knl")
        set(NO_REDZONE ON PARENT_SCOPE)
        set(CPU_ARCH_FLAGS "-march=knl" PARENT_SCOPE)

    # Skylake
    elseif("${architecture}" STREQUAL "skx")
        set(NO_REDZONE ON PARENT_SCOPE)
        set(CPU_ARCH_FLAGS "-march=skylake-avx512" PARENT_SCOPE)

    # Naples (Zen)
    elseif("${architecture}" STREQUAL "naples")
        set(NO_REDZONE ON PARENT_SCOPE)
        set(CPU_ARCH_FLAGS "-march=znver1" PARENT_SCOPE)

    # Rome (Zen 2)
    elseif("${architecture}" STREQUAL "rome")
        set(NO_REDZONE ON PARENT_SCOPE)
        set(CPU_ARCH_FLAGS "-march=znver2" PARENT_SCOPE)

    # Milan (Zen 3)
    elseif("${architecture}" STREQUAL "milan")
        set(NO_REDZONE ON PARENT_SCOPE)
        set(CPU_ARCH_FLAGS "-march=znver3" PARENT_SCOPE)

    # Bergamo (Zen 4c)
    elseif("${architecture}" STREQUAL "bergamo")
        set(NO_REDZONE ON PARENT_SCOPE)
        set(CPU_ARCH_FLAGS "-march=znver4c" PARENT_SCOPE)

    # ThunderX2
    elseif("${architecture}" STREQUAL "thunderx2t99")
        set(NO_REDZONE ON PARENT_SCOPE)
        set(CPU_ARCH_FLAGS "-march=thunderx2t99" PARENT_SCOPE)

    # POWER9
    elseif("${architecture}" STREQUAL "power9")
        set(NO_REDZONE ON PARENT_SCOPE)
        set(CPU_ARCH_FLAGS "-mcpu=power9" PARENT_SCOPE)

    # Apple M1
    elseif("${architecture}" STREQUAL "apple-m1")
        set(NO_REDZONE ON PARENT_SCOPE)
        set(CPU_ARCH_FLAGS "-mcpu=apple-m1" PARENT_SCOPE)

    # Apple M2
    elseif("${architecture}" STREQUAL "apple-m2")
        set(NO_REDZONE ON PARENT_SCOPE)
        set(CPU_ARCH_FLAGS "-mcpu=apple-m2" PARENT_SCOPE)

    # ARM A64FX
    elseif("${architecture}" STREQUAL "a64fx")
        set(NO_REDZONE ON PARENT_SCOPE)
        set(CPU_ARCH_FLAGS "-march=armv8.2-a+sve" PARENT_SCOPE)

    # ARM Neon
    elseif("${architecture}" STREQUAL "neon")
        set(NO_REDZONE ON PARENT_SCOPE)
        set(CPU_ARCH_FLAGS "-march=armv8-a+neon" PARENT_SCOPE)

    else()
        set(CPU_ARCH_FLAGS "-march=noarch" PARENT_SCOPE)
    endif()

endfunction()
