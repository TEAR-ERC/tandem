set(DOMAIN_DIMENSION 2 CACHE STRING "Dimension of the domain")
set(POLYNOMIAL_DEGREE 2 CACHE STRING "Polynomial degree")
set(MIN_QUADRATURE_ORDER "auto" CACHE STRING "Minimum order of quadrature rule")

if(NOT ${POLYNOMIAL_DEGREE} GREATER 0)
    message(FATAL_ERROR "Polynomial degree must be integer and greater 0.")
endif()
