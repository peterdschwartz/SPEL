import csv
from . import assets


def query_statement(table_name, **parm_list):
    
    # table = TABLE_NAME_LOOKUP[parm_list[0]]
    table = assets.TABLE_NAME_LOOKUP[table_name]

    for parm in parm_list:
        table[parm] = parm_list[parm]
      
   
    statement = f"SELECT * FROM {table_name} where "

    for key in table.keys():
        
        statement += f"{key} like '%'" if not table[key] else f"{key}='{table[key]}'"
        
        if key != list(table.keys())[-1]:
            statement += " and "
            
    return statement
       
       
'''
def insert(cur, conn, table_name, csv_file, join=False, ref_table=None, ref_cols=[], dep_col_names =[], dep_cols=[]):
    """
    MYSQL insert code generator
    
    cur, conn: obtained from connect.connet()
    table_name: name of table inserting into 
    csv_file:  csv file where info is read from. NOTE: header must be same order of table
    join:      joining info from another table. If True, next few parameters are required
    ref_table: reference table, str
    ref_cols:  reference columns, same dimension as table_name and order as dep_col_names
    dep_col_names: dependent column names, same dimesions as table_name and order as dep_cols       
    dep_cols:   dependent columns, list of 0/1's same dimension as table_name. 
               1 if column refences another table else 0
    
    
    """
    with open(csv_file, "r") as file:
        reader = csv.reader(file)
        cols = []
        val_types = ["%s"]
        header = True
        for line in reader:
            
            # grab header info from first line
            if header:
                cols = line
                length = len(cols)
                cols = ", ".join(['`' + col.strip() + '`' for col in cols])
                
                val_types = ", ".join(val_types * length)
                header = False
                continue
            STATEMENT = f"INSERT INTO {table_name} ({cols}) "

            if not join :
                
                
                STATEMENT += f"VALUES ({line})"

            else:
                m = []
                pres = []
                clause = []
                x = 0
                for col in range(len(ref_cols)):
                    if dep_cols[col]:
                        m.append(f"m{col}.{ref_cols[col]}")
                        pres.append(f"m{col}.{dep_col_names[col]}")
                        
                        clause.append(f"{ref_table} m{col}")
                        if col % 2 !=0: x+=1
                        
                    else:
                        m.append(f"'{line[col].strip()}'")
                        pres.append(f"'{line[col].strip()}'")
            
                
                        
                ifs = [f"{m[i]}='{line[i]}'" if dep_cols[i] else "" for i in range(len(m))]
           
                lol = ""
                for i in range(len(ifs)):
                    lol += ifs[i]
                    if i < x or ifs[i] != "": 
                        lol += ' and '
                    
                    
                STATEMENT += f"select {', '.join(pres)} from {' join '.join(clause)} where {lol}"
            # print(STATEMENT)
                
            # cur.execute(STATEMENT)
            # conn.commit()
            # insert into table
'''


"""
insert into table0 (cols)
select m1.name, m2.name, m3.name -> 
from table1 m1
join table2 m2
join table3 m3
where m1.id = 1 and
and m2.id = 2 
and m3 = 3 

tables       = [table1, table2, table3]
select_table = [name1, name2, name3]
conditions =   [id1, id2, id3 ]



"""

def insert(cur, 
           conn, 
           table_name, 
           csv_file, 
           join=False, 
           ref_tables=[], 
           selections=[], 
           conditions=[],
           dependency=[],
           extra=[]):
    failed_row = []
    cols = []
    header = True
    length = len(ref_tables)
    isStr = [isinstance(table, str) for table in ref_tables]

    with open(csv_file, "r") as file:
        reader = csv.reader(file)
        val_types = ["%s"]
       
        for line in reader:
            
            # grab header info from first line
            if header:
                cols = line
                cols = ", ".join(['`' + col.strip() + '`' for col in cols])
                
                # val_types = ", ".join(val_types * length)
                header = False
                continue
            STATEMENT = f"INSERT INTO {table_name} ({cols}) "

            if not join :
                STATEMENT += f"VALUES ({line})"
            else:
           
                prefix = [f"m{i}" if isStr[i] else line[ref_tables[i]] for i in range(length)]
                selects = [f"{prefix[i]}.{selections[i]}" if isStr[i] else f"'{prefix[i]}'" for i in range(length)]
                join_tables = [f"{ref_tables[i]} {prefix[i]}" for i in range(length) if isStr[i]]
                intersections = [f"{prefix[i]}.{conditions[i]}='{line[dependency[i]]}'" for i in range(length) if isStr[i] and dependency[i] is not None]
             
                STATEMENT += f"SELECT {', '.join(selects)} FROM {' join '.join(join_tables)} WHERE {' and '.join(intersections)}"
                if extra: STATEMENT += " and " + ' and '.join(extra)
                check_statement = f"SELECT COUNT(*) FROM {' join '.join(join_tables)} WHERE {' and '.join(intersections)}"
                
                cur.execute(check_statement)
                result = cur.fetchone()
                
                if result[0] == 0:
                    # If the row doesn't meet the JOIN conditions, track it and skip insertion
                    failed_row.append(line)
                    continue
                # else:
            # print(STATEMENT)
            cur.execute(STATEMENT)
            conn.commit()
    for i in failed_row:
        print(i)
        
  
