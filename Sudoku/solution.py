rows = 'ABCDEFGHI'
cols = '123456789'

assignments = []

def cross(A, B):
    """Cross product of elements in A and elements in B."""
    return [a+b for a in A for b in B]
def dot(A,B):
    """Dot product of elements in A and elements in B"""
    return [A[i]+B[i] for i,_ in enumerate(A)]

"""Generates useful constants that allow easier handling of data"""
boxes = cross(rows, cols)
row_units = [cross(r, cols) for r in rows]
column_units = [cross(rows, c) for c in cols]
square_units = [cross(rs, cs) for rs in ('ABC','DEF','GHI') for cs in ('123','456','789')]
diag_units = [dot(rows,cols), dot(rows, cols[::-1])]
unitlist = row_units + column_units + square_units + diag_units
units = dict((s, [u for u in unitlist if s in u]) for s in boxes)
peers = dict((s, set(sum(units[s],[]))-set([s])) for s in boxes)

def assign_value(values, box, value):
    """
    Please use this function to update your values dictionary!
    Assigns a value to a given box. If it updates the board record it.
    """
    values[box] = value
    if len(value) == 1:
        assignments.append(values.copy())
    return values

def naked_twins(values):
    """Eliminate values using the naked twins strategy.
    Args:
        values(dict): a dictionary of the form {'box_name': '123456789', ...}
    Returns:
        the values dictionary with the naked twins eliminated from peers.
    """
    from collections import deque, Counter

    q = deque()
    # Find all instances of naked twins
    for unit in unitlist:
        vals = [ values[box] for box in unit if len(values[box])==2]
        counter = Counter(vals)
        for val, cnt in counter.items():
            if cnt != 2: continue
            twin_bxs = [box for box in unit if values[box] == val]
            q.append(set(twin_bxs))

    new_vals = values.copy()
    # Eliminate the naked twins as possibilities for their peers
    while(len(q)):
        t1, t2 = q.pop()
        shared_peers = peers[t1] & peers[t2]
        for peer in shared_peers:
            new_val = values[peer].replace(values[t1][0],'').replace(values[t1][1],'')
            new_vals = assign_value(new_vals, peer, new_val)
    return new_vals

def naked_twins_recursive(values):
    """Eliminate values using the naked twins strategy.
    Eliminates until no other pair of naked twins are found.
    Args:
        values(dict): a dictionary of the form {'box_name': '123456789', ...}
    Returns:
        the values dictionary with the naked twins eliminated from peers.
    """
    from collections import deque, Counter

    q = deque()
    # Find all instances of naked twins
    for unit in unitlist:
        vals = [ values[box] for box in unit if len(values[box])==2]
        counter = Counter(vals)
        for val, cnt in counter.items():
            if cnt != 2: continue
            twin_bxs = [box for box in unit if values[box] == val]
            if(set(twin_bxs) not in q): q.append(set(twin_bxs))
    # Eliminate the naked twins as possibilities for their peers
    while(len(q)):
        t1, t2 = q.pop()
        shared_peers = peers[t1] & peers[t2]
        removed = deque()
        for peer in shared_peers:
            old_len = len(values[peer])
            new_val = values[peer].replace(values[t1][0],'').replace(values[t1][1],'')
            values = assign_value(values, peer, new_val)
            new_len = len(values[peer])
            # If new possible twin pair
            if(new_len == 2 and old_len != new_len):
                removed.append(peer)
        # Find newly created twin pairs
        for rmvd_box in removed:
            for unit in units[rmvd_box]:
                twin_bxs = [box for box in unit if values[box] == values[rmvd_box]]
                if len(twin_bxs)==2: q.append(set(twin_bxs))
    return values


def grid_values(grid):
    "Convert grid into a dict of {square: char} with '123456789' for empties."
    d = dict(zip(boxes, grid))
    for key in d.keys():
        if d[key] == '.': d[key] = '123456789'
    return d

def display(values):
    "Display these values as a 2-D grid."
    width = 1+max(len(values[s]) for s in boxes)
    line = '+'.join(['-'*(width*3)]*3)
    for r in rows:
        print(''.join(values[r+c].center(width)+('|' if c in '36' else '')
                      for c in cols))
        if r in 'CF': print(line)
    print("")

def eliminate(values):
    "Removes the possibiilties of using peer solved digits in unsolved boxes"
    solved = [ (key,value) for key, value in values.items() if len(value)==1]
    for box, val in solved:
        for peer in peers[box]:
            values = assign_value(values, peer, values[peer].replace(val,''))
    return values

def only_choice(values):
    "Assigns unique (non-found in peers) digit to unsolved box."
    for unit in unitlist:
        for digit in cols:
            contain = [box for box in unit if digit in values[box]]
            if len(contain) == 1: values = assign_value(values,contain[0],digit)
    return values

def reduce_puzzle(values):
    "Iteratively applies techniques to solve a puzzle until finished or invalid."
    solved_values = [box for box in values.keys() if len(values[box]) == 1]
    stalled = False
    while not stalled:
        # Check how many boxes have a determined value
        solved_values_before = len([box for box in values.keys() if len(values[box]) == 1])
        # Use the Naked Twins Strategy
        values = naked_twins(values)
        # Use the Eliminate
        values = eliminate(values)
        # Use the Only Choice Strategy
        values = only_choice(values)
        # Check how many boxes have a determined value, to compare
        solved_values_after = len([box for box in values.keys() if len(values[box]) == 1])
        # If no new values were added, stop the loop.
        stalled = solved_values_before == solved_values_after
        # Sanity check, return False if there is a box with zero available values:
        if len([box for box in values.keys() if len(values[box]) == 0]):
            return False
    return values

def solve(grid):
    "Using depth-first search and propagation, create a search tree and solve the sudoku."
    # First, reduce the puzzle using the previous function
    if(type(grid) is not dict): grid = grid_values(grid)
    values = grid
    values = reduce_puzzle(values)
    if values is False:
        return False
    if all([len(values[box])==1 for box in boxes]):
        return values

    # Chose one of the unfilled square s with the fewest possibilities
    _,min_box = min( [(len(values[b]),b) for b in boxes if len(values[b])>1] )

    # Now use recursion to solve each one of the resulting sudokus, and if one returns a value (not False), return that answer!
    for digit in values[min_box]:
        values_copy = values.copy()
        values_copy = assign_value(values_copy,min_box,digit)
        trial = solve(values_copy)
        if trial is not False:
            return trial
    return False

def search(values):
    "TODO: To be implemented. Never needed."
    pass


# --------------------

diag_sudoku_grid = '2.............62....1....7...6..8...3...9...7...6..4...4....8....52.............3'
display(solve(grid_values(diag_sudoku_grid)))

try:
    from visualize import visualize_assignments
    visualize_assignments(assignments)
except:
    print('We could not visualize your board due to a pygame issue. Not a problem! It is not a requirement.')
