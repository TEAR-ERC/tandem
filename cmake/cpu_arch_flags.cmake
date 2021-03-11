function(get_arch_flags architecture compiler)
    set(NO_REDZONE ON PARENT_SCOPE)

    # Sandy Bridge
    if("${architecture}" STREQUAL "snb")
        set(CPU_ARCH_FLAGS "-march=sandybridge" PARENT_SCOPE)
    
    # Haswell
    elseif("${architecture}" STREQUAL "hsw")
        set(CPU_ARCH_FLAGS "-march=haswell" PARENT_SCOPE)

    # Skylake
    elseif("${architecture}" STREQUAL "skl")
        set(CPU_ARCH_FLAGS "-march=skylake" PARENT_SCOPE)
    
    # Skylake X
    elseif("${architecture}" STREQUAL "skx")
        set(CPU_ARCH_FLAGS "-march=skylake-avx512" PARENT_SCOPE)

    # Naples (Zen)
    elseif("${architecture}" STREQUAL "naples")
        set(CPU_ARCH_FLAGS "-march=znver1" PARENT_SCOPE)

    # Rome (Zen 2)
    elseif("${architecture}" STREQUAL "rome")
        set(CPU_ARCH_FLAGS "-march=znver2" PARENT_SCOPE)

    else()
        set(CPU_ARCH_FLAGS "-march=native" PARENT_SCOPE)

    endif()

endfunction()

