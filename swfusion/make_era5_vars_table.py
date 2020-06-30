import copy
import math
import os


def main():
    try:
        txt_path = input('Path of the text file recording ERA5 variables: ')
        cols_num = int(input('Number of columns: '))
        out_path = input('Path of edited text file: ')
        with open(txt_path, 'r') as f:
            txt_lines = f.readlines()

        new_txt_lines = []
        rows_num = []
        vars_num = len(txt_lines)
        tmp_cols_num = copy.copy(cols_num)
        while vars_num:
            row_num_of_this_col = int(math.ceil(vars_num / tmp_cols_num))
            rows_num.append(row_num_of_this_col)
            vars_num -= row_num_of_this_col
            tmp_cols_num -= 1

        table_cells = []
        for i in range(max(rows_num)):
            row = []
            # Multiple 2 to store number of variable
            for j in range(2 * cols_num):
                row.append('')

            table_cells.append(row)

        vars_num = len(txt_lines)
        new_line = ''
        # Only refer to column index of variable name
        col_idx = -1
        for txt_line_idx, line in enumerate(txt_lines):
            num_of_var = txt_line_idx + 1
            # Check which column does the variable fall into
            rows_num_sum = 0
            for idx, num in enumerate(rows_num):

                if (txt_line_idx >= rows_num_sum
                        and txt_line_idx < rows_num_sum + num):
                    if txt_line_idx == 29:
                        pass
                    var_cell_col_idx = 2 * idx + 1

                    if var_cell_col_idx > 2 * 1 - 1:
                        tmp_row_idx = copy.copy(txt_line_idx)
                        for row_num_of_col in rows_num[:idx]:
                            tmp_row_idx -= row_num_of_col
                        var_cell_row_idx = tmp_row_idx
                    else:
                        var_cell_row_idx = txt_line_idx

                    break

                rows_num_sum += num

            if not (var_cell_col_idx - 1):
                var_num_cell = f'{txt_line_idx+1}\t'
            else:
                var_num_cell = f'& {txt_line_idx+1}\t'
            table_cells[var_cell_row_idx][
                var_cell_col_idx - 1] =  var_num_cell

            if var_cell_col_idx != 2 * cols_num - 1:
                var_cell = line.replace('\n', '\t')
            else:
                var_cell = line.replace('\n', '\\\\') + '\n'

            table_cells[var_cell_row_idx][
                var_cell_col_idx] =  var_cell

        table_lines = []
        for row in table_cells:
            table_lines.append(''.join(row))

        with open(out_path, 'w') as f:
            f.writelines(table_lines)
    except Exception as msg:
        breakpoint()
        exit(msg)

if __name__ == '__main__':
    main()
