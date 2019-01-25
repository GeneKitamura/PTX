import warnings

def suppress_warnings():
    warnings.filterwarnings('ignore', message='numpy.dtype size changed')
    warnings.filterwarnings('ignore', message='numpy.ufunc size changed')

def check_list():
    todo_list = []
    #todo_list.append("")
    if len(todo_list) != 0:
        print('\nTODO list: ')
    for i, j in enumerate(todo_list):
        print('Item #{}: {}'.format(i + 1, j))

def bug_list():
    todo_list = []
    #todo_list.append("TF bug: Cannot use new map func on datasets created after sess started")
    if len(todo_list) != 0:
        print('\nBug list: ')
    for i, j in enumerate(todo_list):
        print('Bug #{}: {}'.format(i + 1, j))