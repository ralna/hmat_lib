Installation
************

At this stage, the only way to install ``hmat_lib`` is from source:

From Source
===========

CMake
-----

This is the preferred method of installing ``hmat_lib``::

    git clone https://github.com/ralna/hmat_lib.git
    cd hmat_lib
    mkdir build && cd build
    cmake ..
    make

Configuration
^^^^^^^^^^^^^

This way, the build process can be configured in a couple ways - these are the
provided CMake options:

.. confval:: BUILD_SHARED_LIBS
   :type: ``bool``
   :default: ON

   Controls whether ``hmat_lib`` is built as a shared or static library. If
   ``ON``, a shared library is built.

.. confval:: ENABLE_OPENMP
   :type: ``bool``
   :default: ON

   Enables OpenMP - when ``ON``, searches for OpenMP and, if found, compiles
   ``hmat_lib`` with OpenMP.

.. confval:: ASAN
   :type: ``bool``
   :default: OFF

   Enables address sanitisation. If ``ON``, compiles ``hmat_lib`` with the
   following compiler features (depending on compiler):

   * Address Sanitisation (GCC, Clang, AppleClang, MSVC)
   * Leak Sanitisation (GCC, Clang, AppleClang)
   * Undefined Sanitisation (GCC, Clang)

.. confval:: BUILD_TESTING
   :type: ``bool``
   :default: OFF

   Enables the building of tests. If ``ON``, the tests are built in addition
   to ``hmat_lib``.

.. confval:: BUILD_OPENMP_TESTS
   :type: ``bool``
   :default: ON

   Enables the building of OpenMP tests. If ``ON``, additional tests that test
   the OpenMP implementation will be built. Only used if 
   :confval:`BUILD_TESTING` is ``ON`` and OpenMP has been found.

.. confval:: BUILD_REAL_DATA_TESTS
   :type: ``bool``
   :default: OFF

   Enables the building of tests that use real-world data. If ``ON``, an 
   additional integration test that tests the top-level functions on a 
   real-world Hamiltonian matrix will be built. Only used if
   :confval:`BUILD_TESTING` is ``ON``. Requires the matrix to be found as a
   text file ``hmat_lib/tests/data/H`` (not included via git).

.. confval:: REAL_DATA_PRINT_S
   :type: ``bool``
   :default: OFF

   Enables the logging of the ranks and fill percentages in the real-world
   data test. Only used when :confval:`BUILD_REAL_DATA_TESTS` is ``ON``.

