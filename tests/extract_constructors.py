import os

import clang
import clang.cindex


TEST_DIR = os.path.dirname(os.path.abspath(__file__))
HOME_DIR = os.path.dirname(TEST_DIR)
SRC_DIR = os.path.join(HOME_DIR, 'src')
INCLUDE_DIR = os.path.join(HOME_DIR, 'include')

TEST_SRC_DIR = os.path.join(TEST_DIR, 'src')
TEST_INCLUDE_DIR = os.path.join(TEST_DIR, 'include')


SRC_FILE = os.path.join(SRC_DIR, 'constructors.c')
INCLUDE_FILE = os.path.join(INCLUDE_DIR, 'tree.h')


EXTRACT_FUNCS = [
    'dense_to_tree_hodlr',
    'compress_off_diagonal',
    'free_tree_hodlr',
    'free_partial_tree_hodlr',
    'free_tree_data',
    'allocate_tree',
]


def change_func_name(name):
    return f'{name}_cr'


if __name__ == '__main__':
    locations = []
    declarations = []
    func_names = []

    idx = clang.cindex.Index.create()
    ast = idx.parse(SRC_FILE)#, args=f'-isystem {INCLUDE_DIR}')
    for c in ast.cursor.walk_preorder():
        if c.location.file is None or c.location.file.name != SRC_FILE:
            continue

        if c.kind == clang.cindex.CursorKind.FUNCTION_DECL and c.spelling in EXTRACT_FUNCS:
            locations.append(c.extent)
            declarations.append(c.type.spelling.split('(')[0] + c.displayname)
            func_names.append(c.spelling)

    with open(os.path.join(TEST_INCLUDE_DIR, 'tree_stubs.h'), 'w') as f:
        f.write('#ifndef TREE_STUBS_H\n#define TREE_STUBS_H\n\n')
        f.write('#include "../../include/tree.h"\n\n')

        for declaration, name in zip(declarations, func_names):
            declaration = declaration.replace('restrict', '') \
                                     .replace(name, change_func_name(name)) \
                                     .replace('static', '')
            f.write(f'{declaration};\n\n')
        f.write('#endif\n')

    with open(SRC_FILE, 'r') as f:
        src = f.readlines()
        print(len(src))

    out_c = os.path.join(TEST_SRC_DIR, 'tree_stubs.c')
    with open(out_c, 'w') as f:
        f.write('#include <stdio.h>\n#include <math.h>\n\n')
        f.write('#include <criterion/criterion.h>\n#include <criterion/parameterized.h>\n\n')
        f.write('#include "../include/tree_stubs.h"\n\n')
        f.write('#include "../../include/error.h"\n#include "../../include/lapack_wrapper.h"\n'
                '#include "../../include/tree.h"\n\n\n')
        for loc, name in zip(locations, func_names):
            source = src[loc.start.line-1:loc.end.line+1]
            source[0] = source[0].replace('static', '')
            source = ''.join(source).replace('malloc', 'cr_malloc').replace('free(', 'cr_free(')

            for changed in EXTRACT_FUNCS:
                source = source.replace(changed, change_func_name(changed))
            f.write(source + '\n\n')

