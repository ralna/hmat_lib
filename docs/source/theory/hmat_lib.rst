hmat_lib Library
================

How does the library work?
--------------------------

The below pages explain *how* ``hmat_lib`` works, describing the data 
structures and how they relate to :term:`HODLR`. For, instead, *why* the 
library works the way it works, read on in the 
:ref:`next section<design-decisions>`.

.. toctree::
   :maxdepth: 1
   :glob:

   library/*


.. _design-decisions:

Design decisions
----------------

``hmat_lib`` was designed with HODLR-HODLR matrix-matrix multiplication in 
mind. In the process of its implementation, several design choices have been
made that influenced the process. These and their reasons are detailed here:

HODLR as a perfectly balanced tree
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The HODLR matrix in ``hmat_lib`` is represented via a perfectly balanced 
:term:`tree`. For more details on the data structure and how it relates to 
HODLR matrices, see :doc:`library/hodlr` - here we discuss the reasons for 
and consequences of this decision. In this case, the reasons for this 
decision are quite straightforward:

1. Simplicity

   * It is simpler to conceptualise and work with a :term:`tree` that is 
     perfectly balanced (+ fewer things to unit test!) - this way there are no 
     extraneous cases and it is simpler and more efficient to 
     :ref:`iterate over<tree-iteration>`.

2. Practicality

   * In many :doc:`applications<why_hodlr>` when a matrix is converted into
     the HODLR format, the HODLR ends up fairly balanced - the ranks on all
     off-diagonal nodes, both between different levels and within a level,
     tend to be similar. A perfectly balanced :term:`tree` represents such
     arrangement well.

The consequence of this decision is that only uniformly deep :term:`HODLR` 
matrices can be represented in ``hmat_lib``. Fortunately, this is the default,
and if a more complex arrangement ever becomes of interest, it should be 
possible to extend the current framework.


.. _hodlr-always-square:

HODLR matrix is always square
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :c:struct:`TreeHODLR` data structure has been designed to always represent
a square :term:`HODLR` matrix - not only are there no routines for converting
a rectangular matrix into the :term:`HODLR` format, with the current 
``struct``\ s, it is impossible to generate one at all. This is because, 
again, there is limited interest in such an arrangement and would require
:ref:`additional work<diagonal-always-square>`.


.. _diagonal-always-square:

Diagonal blocks are square matrices
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :c:struct:`TreeHODLR` data structure has also been designed so that all 
the :term:`diagonal leaf nodes` store *squre* blocks. This way, the 
:term:`HODLR` always captures the diagonal - which is typically the densest 
region of the kind of matrix well represented by a :term:`HODLR` - using dense
data. As a consequence, however, a rectangular :term:`HODLR` may be difficult 
to represent, though there are currently 
:ref:`no plans to do so<hodlr-always-square>`.


.. _tree-iteration:

HODLR tree traversal uses iteration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For all operations on :c:struct:`TreeHODLR`, it has been decided to iterate
over the :term:`tree` rather than use recursion. This was mostly done because
of concerns for how well recursion would be handled (i.e. potential to run
into stack overflows etc.), but no rigorous tests on the topic were preformed
and so is more of a preference choice.

