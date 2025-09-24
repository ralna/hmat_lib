Iterating Over HODLR Tree
=========================

A :term:`HODLR` :term:`tree` can be iterated through in two directions:

.. contents::
    :backlinks: entry
    :depth: 2
    :local:


Top-down iteration
-------------------

The first direction is to start at the :term:`root node` (``level==0``) and 
then iterate through the :term:`tree` until the end (``level==height-1``).
However, in practice, this can be implemented in two ways:

Using multiple arrays
^^^^^^^^^^^^^^^^^^^^^

.. note::

   This approach has been phased out in ``hmat_lib``.

The conceptually simpler approach is to use one array for storing pointers
to :term:`internal nodes` on one level, and a second array for storing 
pointers for the next level. These can then be swapped between levels:

.. code:: C

   int iter1(struct TreeHODLR *hodlr, struct HODLRInternalNode **q2) {
     struct HODLRInternalNode **q1 = hodlr->work_queue, **temp_ptr = NULL;
     q1[0] = hodlr->root;
     int len_queue = 1;

     for (int level = 0; level < hodlr->height; level++) {
       for (int parent = 0; parent < len_queue; parent++) {
         // Populate new-level array
         q2[2 * parent] = q1[parent]->children[0].internal;
         q2[2 * parent + 1] = q1[parent]->children[3].internal;
       }
       len_queue *= 2;

       // Swap pointers
       temp_ptr = q1;
       q1 = q2;
       q2 = temp_ptr;
     }
   }

In this approach, the first N elements in the ``q1`` array always hold the N
nodes from the lower :term:`level` and the first 2*N elements in the ``q2`` 
array always hold the 2*N nodes from the higher level. After a level is 
iterated over, the ``q2`` pointer is swapped over to ``q1`` to be used as a 
new source, and the original ``q1`` array is reused in ``q2`` as the new 
destination and will be overwritten. At each iteration, either the ``q1`` or 
``q2`` nodes can be used for computations etc. At the end, the ``q1`` array 
stores the highest-level :term:`internal nodes`, which can be utilised in an 
additional loop after the above one, if necessary.

**Pros**

* Simple to understand
* Consecutive entries in arrays being used

**Cons**

* Requires two arrays (twice the memory, though storage is not a significant 
  concern)


Using one array
^^^^^^^^^^^^^^^

.. note::

   This is the preferred top-down iteration approach in ``hmat_lib``, though
   bottom-up looping is preferred when possible.

The more complex approach is to use only one array and use clever indexing
to avoid premature overwriting:

.. code:: C

   int iter2(struct TreeHODLR *hodlr) {
     struct HODLRInternalNode **queue = hodlr->work_queue;
     queue[0] = hodlr->root;

     int len_queue = 1;
     int q_next_node_density = hodlr->len_work_queue;
     int q_current_node_density = q_next_node_density;

     for (int level = 0; level < hodlr->height; level++) {
       // The next level has twice as many nodes
       q_next_node_density /= 2;

       for (int parent = 0; parent < len_queue; parent++) {
         const int idx = parent * q_current_node_density;

         // Populate new-level array
         // Place child 4 halfway between occupied indices
         queue[(2 * parent + 1) * q_next_node_density] = 
           queue[idx]->children[3].internal;

         // Replace parent with child 1
         queue[idx] = queue[parent]->children[0].internal;
       }
       len_queue *= 2;
       q_current_node_density = q_next_node_density;
     }
   }

In this approach, the :term:`nodes` at each :term:`level` are placed 
strategically at indices of ``queue`` such that there is always enough space
between two occupied indices to fit all the descendants of that node. As an
example. the :term:`root node` is placed at index ``0``, its first child then
replaces it at index ``0`` while its fourth child is placed halfway through 
``queue``. The first child of the first child then replaces the first child
at index ``0`` while its fourth child is placed halfway between index ``0``
and the halfway point of ``queue``, etc.

At each iteration, the nodes from ``queue`` can be used either before or after
being updated. At the end, the ``queue`` array stores the highest-level 
:term:`internal nodes`, which can be utilised in an additional loop after the 
above one, if necessary.

**Pros**

* Only requires one array

**Cons**

* More difficult to understand
* Non-consecutive elements of the array are being used (until the last 
  iteration)


Bottom-up iteration
-------------------

The other direction utilises the :c:member:`TreeHODLR.innermost_leaves` array
(``level==height``) to access the highest-level :term:`internal nodes` 
(``level==height-1``) and then iterates up the :term:`tree` until the 
:term:`root node` is reached:

.. code:: C

   int iter3(struct TreeHODLR *hodlr) {
     struct HODLRInternalNode **queue = hodlr->work_queue;
     int n_parent_nodes = hodlr->len_queue;

     for (int parent = 0; parent < n_parent_nodes; parent++) {
       queue[parent] = hodlr->innermost_leaves[2 * parent]->parent;
     }

     for (int level = hodlr->height - 1; level > 0; level--) {
       n_parent_nodes /= 2;

       for (int parent = 0; parent < n_parent_nodes; parent++) {
         queue[parent] = queue[2 * parent]->parent;
       }
     }
   }

In this approach, the work array, ``queue``, is first fully populated with 
with the highest-level :term:`internal nodes`, after which as we iterate over
the :term:`tree`, the nodes on each level take up half as many indices as the
previous, until the :term:`root node` takes up only the first index.

**Pros**

* Only requires one array
* Simple to understand
* Consecutive elements of the array are used

**Cons**

* Requires an additional loop at the beginning, if ``queue`` is not already
  populated
