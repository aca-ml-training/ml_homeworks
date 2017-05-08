from argparse import ArgumentParser
import sys

DESCRIPTION = 'csvcut - selects the specified fields from cvs file'
EXAMPLES = 'csvcut -f 1,2 stat.txt | less -SR'

def keep_fields(ind, row, output_stream):
    """
    Prints the specified fields from .csv file
    
    :param indexes: indexes of the required fields represented as a list of indexes
    :param row: row represented as a list of columns
    :param output_stream: a stream to print the row with required columns
    """
    output_line = ''
    for i, column in enumerate(row):
        if(i in ind):
            if(i == max(ind)):
                output_line += column + ""
            elif(i < max(ind)):
                
                output_line += column  + ','
    
    output_line += '\n'
    output_stream.write(output_line)
    


def main():
    args = parse_args()
    input_stream = open(args.file, 'r') if args.file else sys.stdin
    output_stream = open(args.output_file, 'r') if args.output_file else sys.stdout

    columns = input_stream.readline().strip().split(args.separator)
    first_rows = [columns]
    
    
    
    fields = args.fields.strip().split(args.separator)
    
    if(fields[0].isdigit()):
        fields = list(map(int, fields)) 
        
    for i in range(1):
        first_rows.append(input_stream.readline().strip().split(args.separator))

    indexes = []
    col_counter = 0
    unique_list = list(set(columns))
    unique_index = [columns.index(x) for x in unique_list]
    
    for i,row in enumerate(first_rows):
        if(i == 0):
            for j, column in enumerate(row):
                col_counter += 1
                if(fields[0] == 'all'):
                    indexes.append(j) 
                elif(fields[0] == ''):
                    indexes = []
                elif(isinstance(fields[0], int)):
                    if(j in fields):
                        indexes.append(j)
                elif(isinstance(fields[0], str)):
                    if(column in fields):
                        indexes.append(j)
    
        
                          
        if(args.complement):
            
                
            ind = list(set(range(0,col_counter)) - set(indexes))
            if(args.unique):
                ind = [x for x in unique_index if x in ind]
            keep_fields(ind, row, output_stream)
        else:
            if(args.unique):
                indexes = [x for x in unique_index if x in indexes]
            keep_fields(indexes, row, output_stream)
        
   

    if input_stream != sys.stdin:
        input_stream.close()
    if output_stream != sys.stdout:
        output_stream.close()


def parse_args():
    parser = ArgumentParser(description=DESCRIPTION, epilog=EXAMPLES)
    parser.add_argument('-s', '--separator', type=str, help='Separator to be used', default=',')
    parser.add_argument('-o', '--output_file', type=str, help='Output file. stdout is used by default')
    parser.add_argument('-f', '--fields', type=str, nargs = "?", const = "",help='Fields to keep', default='all')
    parser.add_argument('-c', '--complement', help='Keep complement of the fields', action='store_true')
    parser.add_argument('-u', '--unique', help='No duplicates for fields', action='store_true')
    parser.add_argument('file', nargs='?', help='File to read input from. stdin is used by default')

    args = parser.parse_args()

    return args

main()