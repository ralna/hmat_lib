Glossary
========

.. glossary::

    HODLR
        Hierarchical Off-Diagonal Low-Rank matrix. See 
        :doc:`theory/hodlr_theory` for more information.

    low-rank matrix
    low-rank format
        Matrix representation obtained by approximating a matrix using a 
        truncated singular value decomposition. For more information, see
        :ref:`low-rank-explanation`.

    tree
        Tree-shaped structure used to represent the hierarchical nature of a 
        :term:`HODLR` matrix. Composed of :term:`nodes` connected in a 
        tree-like shape. For more information, see 
        :doc:`theory/library/hodlr`.

    node
    nodes
        Basic unit of a :term:`tree` data structure - multiple nodes connected
        together in a tree shape make up a :term:`tree`. Depending on its 
        type, a node may either hold data or connect to children nodes.

    height
        The number of edges on the longest path from the :term:`root node` to
        the bottommost :term:`leaf node`. Also, the number of :term:`levels` 
        composing a :term:`tree`. E.g., a tree with one :term:`root node`
        and four :term:`children` will have a height of 1.

    level
    levels
        The number of edges on the longest path from the :term:`root node` to
        a particular :term:`node`. Nodes that are the same number of edges 
        away from the :term:`root` are on the same level. E.g., the 
        :term:`root node` is always at ``level==0``, its children are at
        ``level==1``, etc.

    root
    root node
        The topmost :term:`node` of a :term:`tree`, i.e. its beginning. Has no 
        :term:`parent`. In ``hmat_lib``, a root node is always an 
        :term:`internal node`.

    leaf
    leaves
    leaf node
    leaf nodes
        A terminal :term:`node`, i.e. its end. Has no :term:`children`.

    internal node
    internal nodes
        A :term:`node` that has one or more :term:`children`. In ``hmat_lib``,
        an internal node does not store any data. For more information, see
        :ref:`HODLR structure explanation<internal-node-explanation>`

    diagonal node
    diagonal nodes
    diagonal leaf node
    diagonal leaf nodes
        A :term:`leaf node` which represents a dense block on the diagonal
        of the :term:`HODLR` matrix. For more information, see
        :ref:`HODLR structure explanation<diagonal-node-explanation>`

    off-diagonal node
    off-diagonal nodes
    off-diagonal leaf node
    off-diagonal leaf nodes
        A :term:`leaf node` which represents a low-rank block off the diagonal
        of the :term:`HODLR` matrix. For more information, see
        :ref:`HODLR structure explanation<offdiagonal-node-explanation>`

    parent
    parent node
        A :term:`node` that is the ancestor of another :term:`node` (its 
        :term:`child`). In ``hmat_lib``, a parent node is always an 
        :term:`internal node`.

    child
    children
    child node
    child nodes
        A :term:`node` that descends from another :term:`node` (its 
        :term:`parent`)

    subtree
        A :term:`tree` formed by a :term:`node` and all its descendants.



