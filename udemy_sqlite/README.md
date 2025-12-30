
## SQLite Databases | Python Programming: (Build App and API )
- Instructor: Bluelime Learning Solutions

## Section 1: SQLite Database Server Setup

### 1. Introduction

### 2. What is SQLite
- Lite RDBM
- Sqlite features
  - Self-contained
  - Serverless
  - Zero-configuration
  - Transactional
  - ACID (Atomic, Consistent, Isolated, Durable)-compliant
  - Uses dynamic types for tables
  - Allows a single database connection to access multiple files simultaneously
  - Capable of creating in-memory databases

### 3. Download and install SQLite
- https://www.sqlite.org/download.html
- At Ubuntu:
  - For sqldiff: sudo apt install sqlite3-tools

### 4. What is SQLite Studio
- https://sqlitestudio.pl/
- Portable - just unpack. No installation
- Intuitive interface
- Powerful yet light and fast
- Cross-platform
- Exporting to various formats
- Importing data from various formats
- Opensource and free

### 5. Attaching Sample Database
- chinook.db
![db+schema](./db+schema.jpg)

### 6. Connecting to SQLite Database
```bash
$ sqlite3 ./chinook.db 
SQLite version 3.51.0 2025-11-04 19:38:17
Enter ".help" for usage hints.
sqlite> .tables
albums          employees       invoices        playlists     
artists         genres          media_types     tracks        
customers       invoice_items   playlist_track
sqlite> .quit
$
```

### 7. Database Concepts
- Collection of organized information or data stored in a table
- Table
  - Stores information in rows and columns
- Relational databases
  - A database with multiple related tables
- Relational database management system: RDBMS
  - Oracle, MS SQL, ...
- Primary key
  - Uniquely identifies each record in a table
- Foreign key
  - Used to link or reference related tables
- Constraints
  - Used to specify rules for data in a table

### 8. SQLite3 Tool

### 9. Common SQLite3 Commands
```bash
$ sqlite3 dbname.db
sqlite> .tables # shows all table
sqlite> .schema table_name # shows schema of the table
sqlite> .database # shows the current database
main: /home/hpjeon/hw/class/udemy_sqlite/chinook.db r/w
sqlite .help # command helps
sqlite> .output albums.sql
sqlite> SELECT title FROM albums
   ...> ORDER BY title
   ...> LIMIT 24;
sqlite> .quit
$ cat albums.sql
```

### 10. SQLite Dump Command
- .dump # dump the entire database or tables into a screen
- Dump into a new sql file (text)
  - This is NOT csv - consist of SQL commands 
```bash
sqlite> .output ./jeonbx0a.sql
sqlite> .dump
sqlite> .tables
albums          employees       invoices        playlists     
artists         genres          media_types     tracks        
customers       invoice_items   playlist_track
sqlite> .output albums.sql
sqlite> .dump albums # albums table only
```

## Section 2: SQLite Data Manipulation

### 11. SQLite SELECT Statement
```bash
sqlite> SELECT
   ...> 1+2;
3
sqlite> SELECT
   ...> 10  / 3, 3*5;
3|15
```

### 12. Querying all columns in a table
```sql
SELECT *
FROM table_name;
```

### 13. Querying specific columns in a table
```sql
SELECT col1, col2
FROM table_name;
```

### 14. Sorting Data
```sql
SELECT col1, col2
FROM table_name
ORDER BY col1 asc | desc ;
```

### 15. Removing duplicate records
```sql
SELECT DISTINCT col1
FROM table_name;
```

### 16. Filtering Records
```sql
SELECT col_list
FROM table
WHERE search_condition;
-- example:
SELECT name, composer, albumid
FROM tracks
WHERE albumid = 1;
```

### 17. Identifying NULL Values
- A field with no value is referred to as NULL
```sql
SELECT col_list
FROM table_name
WHERE col IS NULL
-- Ex)
SELECT Company
FROM customers
WHERE Company IS NULL;
```

### 18. SQLite Data Types
- MySQL/PostgreSQL use static typing
- SQLite uses dynamic type system
  - SQLite provides five basic data types called storage classes

| Storage class | Description|
|---------------|-------------|
| NULL | Missing or unknown informatino |
| INTEGER | 64-bit signed integer |
| REAL | 64-bit floating-point|
| TEXT | Character data |
| BLOB | Large binary object |

- typeof(): returns storage classes
```bash
sqlite> SELECT
   ...> typeof(100),
   ...> typeof(1.23),
   ...> typeof('hello'),
   ...> typeof(x'1000'),
   ...> typeof(NULL);
integer|real|text|blob|null
```

## Section 3: Creating Database Objects

### 19. SQLite Constraints
- Constraints are used to specify rules for data in a a table
- Constraints can be specified when the table is created with the CREATE TABLE statement, or after the table is created with the ALTER TABLE statement
- Constraints can be column level or table level. Column level constraints apply to a column, and table_level constraints apply to the whole table
- NOT NULL: ensures that a column cannot have a NULL value
- UNIQUE: ensures that all values in a colum are different
- PRIMARY KEY: a combo of NOT NULL and UNIQUE
- FOREIGN KEY: uniquely identifies a row/record in another table
- CHECK: ensures that all values in a column satisfies a specific condition
- DEFAULT: sets a default value for a column when no value is specified
- SQLite CREATE table syntax
```sql
CREATE TABLE [IF NOT EXISTS] [schema_name].table_name (
  col_1 data_type PRIMARY KEY,
  col_2 data_type NOT NULL,
  col_3 data_type DEFAULT 0,
  table_constraint
) [WITHOUT ROWID];
```

### 20. SQLite Create Table Statements - Part 1
- The name `SQLITE` is reserved for internal use
- Crow's foot notation
  - One or zero: `]o+--`
  - One and only one: `]-++--`
  - Zero or Many: `]>o--`
  - One or Many: ``]>+--`

### 21. SQLite Create Table Statements - Part 2
```sql
CREATE TABLE contacts (
 contact_id INTEGER PRIMARY KEY,
 first_name TEXT NOT NULL,
 last_name TEXT NOT NULL,
 email text NOT NULL UNIQUE,
 phone text NOT NULL UNIQUE
 );
CREATE TABLE groups (
  group_id integer PRIMARY KEY,
  name text NOT NULL
); 
CREATE TABLE contact_groups (
  contact_id integer,
  group_id integer,
  PRIMARY KEY (contact_id, group_id),
  FOREIGN KEY (contact_id) REFERENCES contacts (contact_id)
  ON DELETE CASCADE ON UPDATE NO ACTION,
  FOREIGN KEY (group_id) REFERENCES groups (group_id)
  ON DELETE CASCADE ON UPDATE NO ACTION
);
```

### 22. SQLite INSERT INTO Table Statement
- Inserting new rows into a table
```sql
INSERT INTO table1 (col_1, col2, ...)
VALUES ( val1, val2, ..);
-- Ex)
INSERT INTO contacts(first_name, last_name, email, phone)
VALUES('Jonny', 'Cash', 'jonny@cash.com', '123-456-789');
SELECT * FROM contacts;
```  

### 23. SQLite UPDATE Statement
```sql
UPDATE table_name
SET col1=val1, col2=val2
WHERE condition;
-- Ex)
UPDATE contacts
SET last_name= "Johnson"
WHERE contact_id = 2;
--
SELECT * FROM contacts;
```

### 24. SQLite DELETE Statement
```sql
DELETE
FROM table_name
WHERE condition;
-- EX)
DELETE FROM groups
WHERE group_id <2;
--
SELECT * FROM groups;
```

## Section 4: SQLite Operators

### 25. SQLite BETWEEN Operator
- Selects values within a given range
```sql
SELECT col_list
FROM table_name
WHERE col_name BETWEEN val_1 AND val_2;
-- Ex)
SELECT
  invoiceid, billingaddress, total
FROM invoices
WHERE total BETWEEN 14.91 and 18.95
ORDER BY total;  
```

### 26. SQLite IN Operator
- Determines if a value matches in a list of values
- Allows you to specify multiple values inside a WHERE clause
```sql
SELECT col_list
FROM table_name
WHERE col_name IN(val_1, val_2, ...)
-- Ex)
SELECT trackid,name,mediatypeid
FROM tracks
WHERE mediatypeid in (1,2)
ORDER by name ASC;
```

### 27. SQLite LIKE Operator
- Pattern matching
- Wild cards: % and _
  - %: multiple position holder
  - _: single position holder
```sql
SELECT col_list
FROM table_name
WHERE col_1 LIKE pattern;
```

| LIKE oeprator | Description |
|---------------|-------------|
| WHERE name LIKE 'a%' | any values staring with "a" |
| WHERE name LIKE '%a' | any values ending with "a" |
| WHERE name LIKE '%or%' | any values having "or" within |
| WHERE name LIKE '_r%' |  any values having "r" in the second |
| WHERE name LIKE 'a_%_%' | any values starting with "a" and 3 characters |
| WHERE name LIKE 'a%o' | any values starting with "and" and ending with "o"|

```sql
SELECT trackid,name
FROM tracks
WHERE name LIKE 'Wi%S%';
```

### 28. SQLite GLOB Operator
- Pattern matching but case sensitive
- Wild cards: `*` and `?`, `[XYZ]`
```sql
SELECT trackid,name
FROM tracks
WHERE name GLOB 'Man*E???';
```

## Section 5: Setting Up Python Programming Environment

### 29. What is Python

### 30. What is Jupyter Notebook

### 31. Installing Jupyter Notebook Server

### 32. Running Jupyter Server

### 33. Notebook Dashboard

### 34. Jupyter Notebook Interface

### 35. Creating a new notebook

## Section 6: Python Programming Fundamentals

### 36. Expressions

### 37. Statements

### 38. Comments

### 39. Data Types

### 40. Casting Data Types

### 41. Variables

### 42. Python List

### 43. Python Dictionary

### 44. Python Operators

### 45. Python Conditional Statements

### 46. Python Loops

### 47. Python Functions

## Section 7: Building a Database App with SQLite and Python - The Setup

### 48. Installing Python on Windows

### 49. Installing Python on Macs

### 50. Installing Python on Linux

### 51. installing Text Editor

### 52. Installing DB Browser for SQlite

### 53. Create a database and table
```bash
$ sqlite3 ./contacts.db
SQLite version 3.51.0 2025-11-04 19:38:17
Enter ".help" for usage hints.
sqlite> CREATE TABLE contacts_list (
(x1...> id INTEGER PRIMARY KEY,
(x1...> name VARCHAR(255) NOT NULL,
(x1...> email VARCHAR(255) NOT NULL UNIQUE,
(x1...> number INTEGER NOT NULL UNIQUE
(x1...> )
   ...> ;
sqlite> 
```

## Section 8: Building The Application Interface

### 54. What we will create
- A GUI showing data, adding and removing records

### 55. Application design sketch

### 56. What is Tkinter
- Cross platform GUI toolkit

### 57. Creating a logo image
- https://logomakr.com

### 58. Creating a project directory

### 59. Importing tkinterModule
```py
from tkinter import Tk, Button, PhotoImage, Label, LabelFrame, W, E, N, S, Entry, END, StringVar, Scrollbar, Toplevel
from tkinter import ttk
```

### 60. Creating a Python Class
```py
class Contacts:
    def __init__(self):
        self.root = root
if __name__ == '__main__':
    root = Tk()
    root.title('My Contact List')
    root.mainloop()
```

### 61. Adding Widgets : Part 1
- Labels
- Entry
- Buttons
- Treeview
- Scrollbar
- Logo
```py
class Contacts:
    def __init__(self,root):
        self.root = root
        self.create_left_icon()
    def create_left_icon(self):
        photo = PhotoImage(file='icon.gif')
        label = Label(image=photo)
        label.image = photo
        label.grid(row=0, column=0)
if __name__ == '__main__':
    root = Tk()
    root.title('My Contact List')
    application = Contacts(root)
    root.mainloop()
```

### 62. Adding Widgets : Part 2
```py
class Contacts:
    def __init__(self,root):
        self.root = root
        self.create_left_icon()
        self.create_label_frame()
    def create_left_icon(self):
        photo = PhotoImage(file='icon.gif')
        label = Label(image=photo)
        label.image = photo
        label.grid(row=0, column=0)
    def create_label_frame(self):
        labelframe = LabelFrame(self.root, text='Create New Contact', bg="sky blue", font="helvetical 10")
        labelframe.grid(row=0,column=1,padx=8, pady=8, sticky='ew')
        Label(labelframe,text='Name:', bg="green", fg="white").grid(row=1,column=1,sticky=W,pady=2,padx=15)
        self.namefield = Entry(labelframe)
        self.namefield.grid(row=1,column=2,sticky=W,padx=5,pady=2)
        Label(labelframe,text='Email:',bg="brown",fg="white").grid(row=2,column=1,sticky=W,pady=2,padx=15)
        self.emailfield= Entry(labelframe)
        self.emailfield.grid(row=2,column=2,sticky=W,padx=5, pady=2)
        Label(labelframe,text='Number:',bg="black",fg="white").grid(row=3,column=1,sticky=W,pady=2,padx=15)
        self.numfield = Entry(labelframe)
        self.numfield.grid(row=3,column=2,sticky=W,padx=5,pady=2)
        Button(labelframe, text='Add Contact', command="", bg="blue", fg="white").grid(row=4,column=2,sticky=E, padx=5, pady=5)
if __name__ == '__main__':
    root = Tk()
    root.title('My Contact List')
    application = Contacts(root)
    root.mainloop()
```

### 63. Adding Widgets : Part 3
```py
class Contacts:
    def __init__(self,root):
        self.root = root
        self.create_left_icon()
        self.create_label_frame()
        self.create_message_area()
        self.create_tree_vew()
        ttk.style = ttk.Style()
        ttk.style.configure("Treeview",font=('helvetica',10))
        ttk.style.configure("Treeview.Heading", 12,'bold')
...
    def create_message_area(self):
        self.message= Label(text='',fg='red')
        self.message.grid(row=3,column=1,sticky=W)
    def create_tree_view(self):
        self.tree = ttk.Treeview(height=10,columns=("email","number"),style="Treeview")
        self.tree.grid(row=6,column=0,columnspan=3)
        self.tree.heading("#0",text="Name",anchor=W)
        self.tree.heading("email",text='Email Address',anchor=W)
        self.tree.heading("number",text='Contact Number',anchor=W)
```

### 64. Adding Widgets : Part 4
- Adding a scrollbar
- Adding more buttons
- Final version of GUI:
```py
from tkinter import Tk, Button, PhotoImage, Label, LabelFrame, W, E, N, S, Entry, END, StringVar, Scrollbar, Toplevel
from tkinter import ttk
class Contacts:
    def __init__(self,root):
        self.root = root
        self.create_gui()
        ttk.style = ttk.Style()
        ttk.style.configure("Treeview",font=('helvetica',10))
        ttk.style.configure("Treeview.Heading", font=('helvetica',12,'bold'))
    def create_gui(self):
        self.create_left_icon()
        self.create_label_frame()
        self.create_message_area()
        self.create_tree_view()
        self.create_scrollbar()
        self.create_bottom_buttons()
    def create_left_icon(self):
        photo = PhotoImage(file='icon.gif')
        label = Label(image=photo)
        label.image = photo
        label.grid(row=0, column=0)
    def create_label_frame(self):
        labelframe = LabelFrame(self.root, text='Create New Contact', bg="sky blue", font="helvetical 10")
        labelframe.grid(row=0,column=1,padx=8, pady=8, sticky='ew')
        Label(labelframe,text='Name:', bg="green", fg="white").grid(row=1,column=1,sticky=W,pady=2,padx=15)
        self.namefield = Entry(labelframe)
        self.namefield.grid(row=1,column=2,sticky=W,padx=5,pady=2)
        Label(labelframe,text='Email:',bg="brown",fg="white").grid(row=2,column=1,sticky=W,pady=2,padx=15)
        self.emailfield= Entry(labelframe)
        self.emailfield.grid(row=2,column=2,sticky=W,padx=5, pady=2)
        Label(labelframe,text='Number:',bg="black",fg="white").grid(row=3,column=1,sticky=W,pady=2,padx=15)
        self.numfield = Entry(labelframe)
        self.numfield.grid(row=3,column=2,sticky=W,padx=5,pady=2)
        Button(labelframe, text='Add Contact', command="", bg="blue", fg="white").grid(row=4,column=2,sticky=E, padx=5, pady=5)
    def create_message_area(self):
        self.message= Label(text='',fg='red')
        self.message.grid(row=3,column=1,sticky=W)
    def create_tree_view(self):
        self.tree = ttk.Treeview(height=10,columns=("email","number"),style="Treeview")
        self.tree.grid(row=6,column=0,columnspan=3)
        self.tree.heading("#0",text="Name",anchor=W)
        self.tree.heading("email",text='Email Address',anchor=W)
        self.tree.heading("number",text='Contact Number',anchor=W)
    def create_scrollbar(self):
        self.scrollbar = Scrollbar(orient='vertical', command=self.tree.yview)
        self.scrollbar.grid(row=6,column=3,rowspan=10,sticky='sn') # direction from south to north
    def create_bottom_buttons(self):
        Button(text='Delete Selected', command="",bg="red",fg="white").grid(row=8,column=0,sticky=W,pady=10,padx=20)
        Button(text='Modify Selected', command="",bg="purple",fg="white").grid(row=8,column=1,sticky=W)
if __name__ == '__main__':
    root = Tk()
    root.title('My Contact List')
    application = Contacts(root)
    root.mainloop()
```

## Section 9: Connecting App to Database

### 65. Setup database connectivity from Python
```py
import sqlite3 
...
class Contacts:
    db_filename = 'contacts.db'
    ...
    def execute_db_query(self,query,parameters=()):
        with sqlite3.connect(self.db_filename) as conn:
            print(conn)
            print('You have successfully connected to the Database')
            cursor = conn.cursor()
            query_result = cursor.execute(query,parameters)
            conn.commit()
        return query_result
```

### 66. Creating Functions : Part 1
- Add new contacts into database
- Validate the inputs
- Call add new contact to database
- Fetch all records from database and display in treeview
```py
    def add_new_contact(self):
        if self.new_contact_validated():
            query = "INSERT INTO contacts_list VALUES (NULL, ?, ?, ?)"
            parameters = (self.namefield.get(),self.emailfield.get(),self.numfield.get())
            self.execute_db_query(query,parameters)
            self.message['text'] = 'New Contact {} added'.format(self.namefield.get())
            self.namefield.delete(0,END)
            self.emailfield.delete(0,END)
            self.numfield.delete(0,END)
        else:
            self.message['text'] = 'name,email and number cannot be blank'
        self.view_records()
    def new_contacts_validated(self):
        return len(self.namefiled.get()) != 0 and len(self.emailfield.get()) !=0 and len(self.numfield.get()) != 0
    def view_contacts(self):
        items = self.tree.get_children()
        for item in items:
            self.tree.delete(item)
        query = 'SELECT * FROM contacts_list ORDER BY name desc'
        contact_entries = self.execute_db_query(query)
        for row in contact_entries:
            self.tree.insert('',0,text=row[1],values==(row[2],row[3]))
    def on_add_contact_button_clicked(self):
        self.add_new_contact()
```

### 67. Creating Functions : Part 2
- Delete contacts from database
```py
    def delete_contacts(self):
        self.message['text'] = ''
        name = self.tree.item(self.tree.selection())['text']
        query = 'DELETE FROM contacts_lists WHERE name = ?'
        self.execute_db_query(query,(name,))
        self.message('text') = 'Contacts for {} deleted'.format(name)
        self.view_contacts()
    def on_delete_selected_button_clicked(self):
      self.message['text'] = ''
      try:
          self.tree.item(self.tree.selection())['values'][0]
      except IndexError as e:
          self.message['text'] = 'No item selected to delete'
          return
      self.delete_contacts()
```

### 68. Creating Functions : Part 3
- Modify contacts from database
```py
    def open_modify_window(self):
        name = self.tree.item(self.tree.selection())['text']
        old_number = self.tree.item(self.tree.selection())['values'][1]
        self.transient = Toplevel()
        self.transient.title('Update Contact')
        Label(self.transient,text='Name:').grid(row=0,column=1)
        Entry(self.transient,textvariable=StringVar(self.transient,value=name),state='readonly').grid(row=0,column=2)
        Label(self.transient,text='Old_contact Number:').grid(row=1,column=1)
        Entry(self.transient,textvariable=STringVar(self.transient,value=old_number),state='readonly').grid(row=1,column=2)
        Label(self.transient,text='New Phone Number:').grid(row=2,column=1)
        new_phone_number_entry_widget = Entry(self.transient)
        new_phone_number_entry.widget.grid(row=2,column=2)
        Button(self.transient,text='Update Contact', command=lambda: self.update_contacts(new_phone_number_entry.widget.get(), old_number,name)).grid(row=3,column=2,sticky=E)
        self.transient.mainloop()
    def update_contacts(self,newphone,old_phone,name):
        query = 'UPDATE contacts_list SET number=? WHERE number =? AND name=?"
        parameters = (newphone,old_phone, name)
        self.execute_db_query(query,parameters)
        self.transient.destroy()
        self.message['text'] = 'Phone number of {} modified'.format(name)
        self.view_contacts()
    def on_modify_selected_button_clicked(self):
        self.message['text'] = ''
        try:
            self.tree.item(self.tree.selection())['values'][0]
        except IndexError as e:
            self.message['text'] = 'No item selected to modify'
            return
        self.open_modify_window()
```        

### 69. Setting a size for the application window
```py
if __name__ == '__main__':
    root = Tk()
    root.title('My Contact List')
    root.geometry("650x450")
    root.resizable(width=False,height=False)
    application = Contacts(root)
    root.mainloop()
```

### 70. Project Code
```py
# Import what you need from tkinter module
from tkinter import Tk, Button, PhotoImage, Label, LabelFrame, W, E,N,S, Entry, END,StringVar ,Scrollbar,Toplevel
from tkinter import ttk   # Provides access to the Tk themed widgets.
import sqlite3
class Contacts:
    db_filename = 'contacts.db'
    def __init__(self,root):
        self.root = root
        self.create_gui()
        ttk.style = ttk.Style()
        ttk.style.configure("Treeview", font=('helvetica',10))
        ttk.style.configure("Treeview.Heading", font=('helvetica',12, 'bold'))
    def execute_db_query(self, query, parameters=()):
        with sqlite3.connect(self.db_filename) as conn:
            print(conn)
            print('You have successfully connected to the Database')
            cursor = conn.cursor()
            query_result = cursor.execute(query, parameters)
            conn.commit()
        return query_result
    def create_gui(self):
        self.create_left_icon()
        self.create_label_frame()
        self.create_message_area()
        self.create_tree_view()
        self.create_scrollbar()
        self.create_bottom_buttons()
        self.view_contacts()
    def create_left_icon(self):
        photo = PhotoImage(file='icon.gif')
        label = Label(image=photo)
        label.image = photo
        label.grid(row=0, column=0)
    def create_label_frame(self):
        labelframe = LabelFrame(self.root, text='Create New Contact',bg="sky blue",font="helvetica 10")
        labelframe.grid(row=0, column=1, padx=8, pady=8, sticky='ew')
        Label(labelframe, text='Name:',bg="green",fg="white").grid(row=1, column=1, sticky=W, pady=2,padx=15)
        self.namefield = Entry(labelframe)
        self.namefield.grid(row=1, column=2, sticky=W, padx=5, pady=2)
        Label(labelframe, text='Email:',bg="brown",fg="white").grid(row=2, column=1, sticky=W,  pady=2,padx=15)
        self.emailfield = Entry(labelframe)
        self.emailfield.grid(row=2, column=2, sticky=W, padx=5, pady=2)
        Label(labelframe, text='Number:',bg="black",fg="white").grid(row=3, column=1, sticky=W,  pady=2,padx=15)
        self.numfield = Entry(labelframe)
        self.numfield.grid(row=3, column=2, sticky=W, padx=5, pady=2)
        Button(labelframe, text='Add Contact', command=self.on_add_contact_button_clicked,bg="blue",fg="white").grid(row=4, column=2, sticky=E, padx=5, pady=5)
    def create_message_area(self):
        self.message = Label(text='', fg='red')
        self.message.grid(row=3, column=1, sticky=W)
    def create_tree_view(self):
        self.tree = ttk.Treeview(height=10, columns=("email","number"),style='Treeview')
        self.tree.grid(row=6, column=0, columnspan=3)
        self.tree.heading('#0', text='Name', anchor=W)
        self.tree.heading("email", text='Email Address', anchor=W)
        self.tree.heading("number", text='Contact Number', anchor=W)
    def create_scrollbar(self):
        self.scrollbar = Scrollbar(orient='vertical',command=self.tree.yview)
        self.scrollbar.grid(row=6,column=3,rowspan=10,sticky='sn')
    def create_bottom_buttons(self):
        Button(text='Delete Selected', command=self.on_delete_selected_button_clicked,bg="red",fg="white").grid(row=8, column=0, sticky=W,pady=10,padx=20)
        Button(text='Modify Selected', command=self.on_modify_selected_button_clicked,bg="purple",fg="white").grid(row=8, column=1, sticky=W)
    def on_add_contact_button_clicked(self):
        self.add_new_contact()
    def on_delete_selected_button_clicked(self):
        self.message['text'] = ''
        try:
            self.tree.item(self.tree.selection())['values'][0]
        except IndexError as e:
            self.message['text'] = 'No item selected to delete'
            return
        self.delete_contacts()
    def on_modify_selected_button_clicked(self):
        self.message['text'] = ''
        try:
            self.tree.item(self.tree.selection())['values'][0]
        except IndexError as e:
            self.message['text'] = 'No item selected to modify'
            return
        self.open_modify_window()
    def add_new_contact(self):
        if self.new_contacts_validated():
            query = 'INSERT INTO contacts_list VALUES(NULL,?, ?,?)'
            parameters = (self.namefield.get(),self.emailfield.get(), self.numfield.get())
            self.execute_db_query(query, parameters)
            self.message['text'] = 'New Contact {} added'.format(self.namefield.get())
            self.namefield.delete(0, END)
            self.emailfield.delete(0, END)
            self.numfield.delete(0, END)
            self.view_contacts()
        else:
            self.message['text'] = 'name,email and number cannot be blank'
            self.view_contacts()
    def new_contacts_validated(self):
        return len(self.namefield.get()) != 0 and len(self.emailfield.get()) != 0 and len(self.numfield.get()) != 0

    def view_contacts(self):
        items = self.tree.get_children()
        for item in items:
            self.tree.delete(item)
        query = 'SELECT * FROM contacts_list ORDER BY name desc'
        contact_entries = self.execute_db_query(query)
        for row in contact_entries:
                self.tree.insert('', 0, text=row[1], values=(row[2],row[3]))
    def delete_contacts(self):
        self.message['text'] = ''
        name = self.tree.item(self.tree.selection())['text']
        query = 'DELETE FROM contacts_list WHERE name = ?'
        self.execute_db_query(query, (name,))
        self.message['text'] = 'Contacts for {} deleted'.format(name)
        self.view_contacts()
    def open_modify_window(self):
        name = self.tree.item(self.tree.selection())['text']
        old_number = self.tree.item(self.tree.selection())['values'][1]
        self.transient = Toplevel()
        self.transient.title('Update Contact')
        Label(self.transient, text='Name:').grid(row=0, column=1)
        Entry(self.transient, textvariable=StringVar(
            self.transient, value=name), state='readonly').grid(row=0, column=2)
        Label(self.transient, text='Old Contact Number:').grid(row=1, column=1)
        Entry(self.transient, textvariable=StringVar(
            self.transient, value=old_number), state='readonly').grid(row=1, column=2)
        Label(self.transient, text='New Phone Number:').grid(
            row=2, column=1)
        new_phone_number_entry_widget = Entry(self.transient)
        new_phone_number_entry_widget.grid(row=2, column=2)
        Button(self.transient, text='Update Contact', command=lambda: self.update_contacts(
            new_phone_number_entry_widget.get(), old_number, name)).grid(row=3, column=2, sticky=E)
        self.transient.mainloop()
    def update_contacts(self, newphone, old_phone,name):
        query = 'UPDATE contacts_list SET number=? WHERE number =? AND name =?'
        parameters = (newphone, old_phone, name)
        self.execute_db_query(query, parameters)
        self.transient.destroy()
        self.message['text'] = 'Phone number of {} modified'.format(name)
        self.view_contacts()
if __name__ == '__main__':
    root =Tk()
    root.title('My Contact List')
    root.geometry("650x450")
    root.resizable(width=False, height=False)
    application = Contacts(root)
    root.mainloop()
```

## Section 10: Build an API with Python + Django + SQLite

### 71. What is an API
- API: Application Programming Interface

### 72. What is Django
- www.djangoproject.com

### 73. Create a virtual environment
- An isolated Python environment
- virtualenv package required
  - venv
- Steps
  - Create a project directory
  - Cd to the folder
  - python3 -m venv myvenv
  - cd myenv
  - source bin/activate
    - Prompt will change
  - echo $VIRTUAL_ENV # shows the current venv
  - Do some work
  - deactivate

### 74. Install Django

### 75. Install REST Framework

### 76. Create Django Project

### 77. Create Django App

### 78. Register Applications

### 79. Install Corsheaders
- CORS: Cross Origin Resource Sharing
  - Allows client applications to interface with APIs hosted on different domains

### 80. SQLite Database Setup
- sqlie is embeedded in Django

### 81. Applying Django Migrations

### 82. Create a model

### 83. Create and apply new migration

### 84. Create a serializer class

### 85. Start and stop development server

### 86. Create a superuser account

### 87. Create views: Part 1

### 88. Create views: Part 2

### 89. Map views to URLS

### 90. Register Model

### 91. Create Model Objects

### 92. Install Postman

### 93. Test API

### 94. Project Code

