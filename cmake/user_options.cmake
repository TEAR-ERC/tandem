set(DOMAIN_DIMENSION 2 CACHE STRING "Dimension of the domain")
set(POLYNOMIAL_DEGREE 2 CACHE STRING "Polynomial degree")
set(MIN_QUADRATURE_ORDER 0 CACHE STRING "Minimum order of quadrature rule, 0 = automatic")

if(NOT ${POLYNOMIAL_DEGREE} GREATER 0)
    message(FATAL_ERROR "Polynomial degree must be integer and greater 0.")
endif()

if(NOT ${MIN_QUADRATURE_ORDER} GREATER_EQUAL 0)
    message(FATAL_ERROR "Minimum order of quadrature rule must be integer and greater equal 0.")
endif()
