## SQL and PostgreSQL: The complete Developer's Guide
- By Stephen Grider

## 3. Creating Tables
- Use pg-sql.com to get a free posgresql database + run queries
```
CREATE TABLE cities (
  name VARCHAR(50),
  country VARCHAR(50),
  population INTEGER,
  area INTEGER
);
```

## 5. Inserting Data into a Table
```
INSERT INTO cities (name, country, population, area)
VALUES ('Tokyo', 'Japan', 38505000, 8223);
INSERT INTO cities (name, country, population, area)
VALUES ('Delhi', 'India', 28125000, 2240),
('Shanghai',  'China',    22125000, 4015),
('Sao Paulo', 'Brazil',   20935000, 3043);
```

## 6. Retrieving data with select
```
SELECT * FROM cities;
SELECT name, area, name, population FROM cities;
```

## 7. Calculated columns
- Math operators
  - +/-/*//
  - ^ : exponent
  - |/ : square root
  - @ : absolute value
  - % : remainder
```
SELECT name, population/area AS population_density FROM cities;
```

## 10. String operators
- || : joins two strings
- CONCAT() : joins multiple strings
- LOWER() : yields a lower case string
- UPPER() : yields a upper case string
- LENGTH() : yields the number of characters in a string
```
SELECT name || ', ' || country FROM cities;
SELECT
  CONCAT(UPPER(name), ', ' , UPPER(country)) AS location
FROM cities;
```

## 11. Filtering with WHERE
- Comparison operator
  - <> : not equal
  - != : not equal
  - = : equal (not ==)
  - >, <, >=, <=, BETWEEN, NOT IN, IN
```
SELECT name, area FROM cities WHERE area > 4000;
SELECT name, area FROM cities WHERE area BETWEEN 2000 AND 4000;
SELECT name, area FROM cities WHERE name IN ('Delhi', 'Shanghai');
SELECT name, area FROM cities WHERE name NOT IN ('Delhi', 'Shanghai');
SELECT name, area FROM cities WHERE area NOT IN (3043,8223) OR name = 'Delhi';
```

## 18. WHERE with calculations
```
SELECT
  name,
  population / area AS population_density
FROM
  cities
WHERE
  population / area > 6000;
```

## 20. Updating rows
```
UPDATE cities
SET population = 39505000
WHERE name = 'Tokyo';
```

## 21. Deleting rows
```
DELETE FROM cities WHERE name='Tokyo';
```

## 25. A sample database design
- One to many relationship: a user -> photos
- Many to one relationship: photos -> a user

## 28. Primary/Foreign keys
- Primary key : unique identifier
- Foreign key : identifier for other links
- Many side has foreign keys
  - One-to-many or many-to-one or many-to-many

## 30. Auto-generated IDs
```
CREATE TABLE users (
  id SERIAL PRIMARY KEY,
  username VARCHAR(50)
  );
INSERT INTO users (username)
VALUES ('john'), ('Mia'), ('Loui'), ('Coco');
```
- Keyword SERIAL PRIMARY

## 31. Generating Foreign keys
```
CREATE TABLE photos(
  id SERIAL PRIMARY KEY,
  url VARCHAR(200),
  user_id INTEGER REFERENCES users(id)
);
INSERT INTO photos(url, user_id)
VALUES
('http://one.jpg', 4);
```
- Keyword REFERENCES
  - Couples with id of users table
  - Works as a constraint

## 32. Taste of JOIN
```
select url, username FROM photos
JOIN users ON users.id = photos.user_id;
```

## 35. Data constraints
- What if user_id in the photos table doesn't match with the id of users table?
```
INSERT INTO photos(url, user_id)
VALUES ('http://jpg',9999);
```
- Produces:
```
insert or update on table "photos" violates foreign key constraint "photos_user_id_fkey"
```
- NULL works OK
```
INSERT INTO photos(url, user_id)
VALUES ('http://jpg',NULL);
```

## 36-40. Constraints around deletion
- Deleting id 1 from users -> produces dangling user_id in photos
- On delete options
  - ON DELETE RESTRICT (default) -> throws an error
  - ON DELETE NO ACTION -> throws an error
  - ON DELETE CASCADE -> Deletes the coupled rows in photos
  ```
  CREATE TABLE photos (
  id SERIAL PRIMARY KEY,
  url VARCHAR(200),
  user_id INTEGER REFERENCES users(id) ON DELETE CASCADE
  );
  ```
  - ON DELETE SET NULL -> corresponding user_id of photos becomes NULL
  ```
  CREATE TABLE photos (
  id SERIAL PRIMARY KEY,
  url VARCHAR(200),
  user_id INTEGER REFERENCES users(id) ON DELETE SET NULL
  );
  ```
  - ON DELETE SET DEFAULT -> corresponding user_id of photos becomes  a default value, when provided

## 37. Commands for testing
```
CREATE TABLE users (
  id SERIAL PRIMARY KEY,
  username VARCHAR(50)
  );
INSERT INTO users (username)
VALUES ('john'), ('Mia'), ('Loui'), ('Coco');

CREATE TABLE photos (
id SERIAL PRIMARY KEY,
url VARCHAR(200),
user_id INTEGER REFERENCES users(id)
);

INSERT INTO photos (url, user_id)
VALUES
('http:/one.jpg', 4),
('http:/two.jpg', 1),
('http:/25.jpg', 1),
('http:/36.jpg', 1),
('http:/754.jpg', 2),
('http:/35.jpg', 3),
('http:/256.jpg', 4);
```

## 41. Adding some data
- Enclosed 41.dataset

## 42. JOINS and AGGREGATION
- JOINS: merging rows and process data
- AGGREGATIONS: Collective operation such as most, average, least, ...

## 43. JOINING data from other tables
```
SELECT contents, username, users.id, user_id
FROM comments
JOIN users ON users.id = comments.user_id;
```
- When key names are ambiguous (or duplicating) like `id`, use table name with like `users.id`

## 47. More on JOINS
- Using AS
```
SELECT contents, username, users.id, user_id
FROM comments AS c
JOIN users ON users.id = c.user_id;
```

## 48. Missing data in JOINS
- When photos.user_id is NULL
  - The above command will not report the case of NULL id
  - FULL or LEFT or RIGHT JOIN is necessary for mis-matching or NULL cases

## 50. 4 kinds of JOIN
- INNER JOIN
  - Regular `JOIN` implies `INNER JOIN`
  - [photos [] users]
```
SELECT url, username
FROM photos
INNER JOIN users ON users.id = photos.user_id;
```
- LEFT JOIN
  - Same as LEFT OUTER JOIN
  - [[photos] users]
  - When values are not found, NULL is printed
```
SELECT url, username
FROM photos
LEFT JOIN users ON users.id = photos.user_id;
```
- RIGHT JOIN
  - Same as RIGHT OUTER JOIN
  - [photos [users]]
```
SELECT url, username
FROM photos
RIGHT JOIN users ON users.id = photos.user_id;
```
- FULL JOIN
  - Same as FULL OUTER JOIN
  - [[photos users]]
```
SELECT url, username
FROM photos
FULL JOIN users ON users.id = photos.user_id;
```

## 55. WHERE with JOIN
```
SELECT url, contents
FROM COMMENTS
JOIN photos ON photos.id = comments.photo_id
WHERE photos.user_id = comments.user_id;
```

## 56. Multiple JOINs
- Merging tables of comments, photos, and users
```
SELECT url, contents, username
FROM COMMENTS
JOIN photos ON photos.id = comments.photo_id
JOIN users ON users.id = comments.user_id
AND users.id = photos.user_id;
```

## 60. GROUP BY
- Finding unique values
```
SELECT user_id
FROM comments
GROUP BY user_id;
```
  - SELECT columns must be the column used by GROUP BY

## 61. Aggregates
- A following command throws an error as `id` is not assigned from GROUP BY
```
SELECT id
FROM comments
GROUP BY user_id;
```
- A following command works as a aggregate function
```
SELECT user_id, max(id)
FROM comments
GROUP BY user_id;
```

## 63. COUNT() for NULL
- COUNT() doesn't count NULL
```
select user_id, COUNT(user_id)
FROM comments
GROUP BY user_id;
```
  - No results of NULL data
- Use COUNT(*) then the cases of NULL are counted
```
select user_id, COUNT(*)
FROM comments
GROUP BY user_id;
```

## 68. HAVING filter
- Similar to WHERE but working on GROUP BY
```
SELECT user_id, COUNT(*)
FROM comments
WHERE photo_id <3
GROUP BY user_id
HAVING COUNT(*) > 2;
```

## 73. Coding exercise
```
SELECT manufacturer, SUM(price*units_sold)
FROM phones
GROUP BY manufacturer
HAVING SUM(price*units_sold) > 2000000;
```

## 74. New dataset
- Enclosed 74.dataset

## 80. The basics of sorting
```
SELECT * FROM products ORDER BY price;
SELECT * FROM products ORDER BY price ASC;
SELECT * FROM products ORDER BY price DESC;
```

## 81. Multiple sorting
```
SELECT * FROM products ORDER BY price, weight ASC;
SELECT * FROM products ORDER BY price ASC, weight DESC;
```

## 82. Offset and Limit
```
SELECT * FROM users OFFSET 40 LIMIT 5;
```
- 40 strides from the start and prints 5 rows only

## 85-86. Union
```
(select * FROM products ORDER BY price DESC LIMIT 4)
UNION
(select * FROM products ORDER BY price/weight DESC LIMIT 4);
```
- UNION: Duplicated rows are printed only once
```
(select * FROM products ORDER BY price DESC LIMIT 4)
UNION ALL
(select * FROM products ORDER BY price/weight DESC LIMIT 4);
```
- UNION ALL: Duplicated items will be printed multiple times
- Columns of each union pair must be matching

## 87. Intersect
- INTERSECT: prints common rows only. Duplicated rows are removed
- INTERSECT ALL: prints common rows only
- Q: ALL is useful? Does duplicated row exist?

## 88. Except
- EXCEPT: Removes rows from the 2nd query. Duplicated rows are removed
- EXCEPT ALL: Removes rows from the 2nd query
- Q: ALL is useful? Does duplicated row exist?

## 91. Subquery
- Ex) Find the more expensive items than all products in the Toys department
  - 1) Query the MAX price in the Toys department
  - 2) Query name and price using the criterion
```
SELECT name, price
FROM products
WHERE price > (
  SELECT MAX(price) FROM products WHERE department = 'Toys'
  );
```

## 93. Subqueries in SELECT
- Subquery in SELECT MUST return a single column data
```
SELECT name, price, (SELECT price FROM products WHERE id=3)
FROM products
WHERE price >867;
```
- Yields:
```
name	                    price	price
Practical Fresh Shirt	    10	  10
Incredible Granite Mouse	10	  10
```
  - First price column data is wrong. This is due to the confusion of the sql engine in pg-sql.com. To fix, use AS for sql engine not to be confused
```
SELECT name, price, (SELECT price FROM products WHERE id=3) AS new_tag
FROM products
WHERE price >867;
```
- Now yields:
```
name	                    price	new_tag
Practical Fresh Shirt	    876	  10
Incredible Granite Mouse	989	  10
```

## 96. Subqueries in FROM
- Must use alias (AS keyword)
- May return multiple columns

## 100. Exercise
- Find the maximum average price of products by departments
```
SELECT MAX(avg_price) AS max_avg_price
FROM ( SELECT AVG(price) AS avg_price
      FROM products GROUP BY department
      ) AS P;
```

## 102. Subqueries with WHERE
```
SELECT id
FROM orders
WHERE product_id IN (
  SELECT id FROM products WHERE price /weight > 50
);
```
- Is equivalent to:
```
SELECT orders.id
FROM orders
JOIN products ON orders.product_id = products.id
AND products.price/products.weight > 50;
```

## 107. Operator with ALL
- > ALL, < ALL, >= ALL, <= ALL, = ALL, <> ALL
  - RHS would be column data
```
SELECT name, department, price
FROM products
WHERE price > (
  SELECT MAX(price)
  FROM products
  WHERE department='Industrial'
);
```
- Is equivalent to:
```
SELECT name, department, price
FROM products
WHERE price > ALL (
  SELECT price
  FROM products
  WHERE department='Industrial'
);
```

## 108. Operator with SOME
- > SOME, < SOME, >= SOME, <= SOME, = SOME, <> SOME
  - RHS would be column data
  - Needs that one data must meet condition at least

## 111. Correlated subqueries
-  Find maximum price item in each department
  - Loop over each department
```
SELECT name, department, price
FROM products AS p1
WHERE p1.price =  (
  SELECT MAX(price)
  FROM products as p2
  WHERE p1.department = p2.department
  );  
```

## 112. Subqueries in SELECT
```
SELECT
(SELECT max(price) FROM phones) AS max_price,
(SELECT min(price) FROM phones) AS min_price,  
(SELECT avg(price) FROM phones) AS avg_price ;
....
max_price	min_price	avg_price
991	      1	        498.7300000000000000
```
- Without FROM in the main SELECT, single values are queried and printed

## 113. SELECT DISTINCT
- Queries unique values using `DISTINCT`
- `SELECT DISTINCT department FROM products;` : queries the unique values in the department column
- `SELECT COUNT(DISTINCT department) FROM products;` : queries the number of unique departments
- `SELECT DISTINCT department, name FROM products;` : queries the unique cases of department and name. Same value might be repeated but same combination is excluded

## 119. GREATEST()
- `SELECT GREATEST(30,10,20);`
  - `SELECT MAX(30,10,20);` doesn't work
```
SELECT name, weight, GREATEST(30, 2* weight)
FROM products;
```
- 3rd column will be the larger value of 30 or 2*weight

## 120. LEAST()
```
SELECT name, weight, LEAST(400, 0.5*price)
FROM products;
```
- 3rd column will be the smaller one of 400 or price/2

## 121. CASE()
```
SELECT
name,
price,
CASE
 WHEN price > 600 THEN 'high'
 WHEN price > 300 THEN 'medium'
 ELSE 'cheap'
END
FROM products;
```
- If the condition of CASE-END is not met, NULL will be printed

## PosgreSQL + pgadmin for Ubuntu
- https://www.tecmint.com/install-postgresql-and-pgadmin-in-ubuntu/
- https://medium.com/coding-blocks/creating-user-database-and-adding-access-on-postgresql-8bfcd2f4a91e
- sudo apt install postgresql postgresql-contrib libpq-dev -y
- sudo apt install pgadmin3
- sudo pg_isready
```
sudo -u postgres psql
postgres=# create database mydb;
postgres=# create user myuser with encrypted password 'XXXX';
postgres=# alter user myuser createdb;
postgres=# grant all privileges on database mydb to myuser;
```
- Open pgadmin3
  - Host is localhost
  - Port 5432
- Ref: https://alvinalexander.com/blog/post/postgresql/log-in-postgresql-database/
- To login: `psql -d mydb -U myuser -h localhost`

- PostgreSQL 12 for Ubuntu 18
  - https://computingforgeeks.com/install-postgresql-12-on-ubuntu/
  - wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | sudo apt-key add -
  - echo "deb http://apt.postgresql.org/pub/repos/apt/ `lsb_release -cs`-pgdg main" |sudo tee  /etc/apt/sources.list.d/pgdg.list
  - sudo apt update
  - sudo apt -y install postgresql-12 postgresql-client-12 pgadmin4
  - systemctl status postgresql.service
  - systemctl is-enabled postgresql
  - Reboot
  - pgadmin4 # may take a couple of minutes
    - ~~Edit /etc/postgresql/12/main/pg_hba.conf and `host ...  md5` as `host ... `, removing md5~~
    - local login using myuser, postgres, and os ID didn't work. psql through `-h localhost` worked OK.
    - sudo service postgresql restart
  - Servers -> right mouse button -> Create new server -> General:Name = localhost -> Connection:Host = localhost
    - Used posgres has the user

## 126. Data type
- smallint, integer, bigint      # int with 16,32, 64 bit
- smallserial, serial, bigserial # unsigned int
- decimal, numeric, real, double precision, float

## 127. Best practice for the data type
- Id column of tables => serial
- Non-floating number => integer
- Bank balance, scientific calculations => numeric
- When round-off error can be tolerated => double precision

## 128. More on number
- `SELECT (2.0);` # psql will store as numeric
- `SELECT (2.0::INTEGER);` # enforce as integer
- `SELECT(1.99999::REAL - 1.99998::REAL);` yields 1.00136e-5
- `SELECT(1.99999::DECIMAL - 1.99998::DECIMAL);` yields 0.00001
- `SELECT(1.99999::NUMERIC - 1.99998::NUMERIC);` yields 0.00001

## 129. Character types
- CHAR(5): Static storage. Spaces are padded if not filled
- VARCHAR: stores a string of any length
- TEXT: stores a string of any length
- VARCHAR(50): Stores up to 50 characters, removing extra characters
- No performance differences on using TEXT or VARCHAR or VARCHAR(#)
  - Finite size might be given to VARCHAR() for certain data check or to avoid memory abuse

## 130. Boolean types
- TRUE:  true,  'yes', 'on',  1, 't', 'y'
- FALSE: false, 'no',  'off', 0, 'f', 'n'
- NULL: null # NULL is BOOLEAN TYPE as well

## 131. DATE
```
SELECT ('1890-11-22'::DATE);
SELECT ('1890 NOV 22'::DATE);
SELECT ('1:33 PM'::TIME);
SELECT ('1:33 PM EST'::TIME WITH TIME ZONE);
SELECT ('NOV-22-2002 1:33 PM EST'::TIMESTAMP WITH TIME ZONE);
```

## 132. INTERVAL
- Duration of time such as day, minute, sec
- `SELECT('1 D 20H 30 M 4 S'::INTERVAL);`
- `SELECT('1 D 20H 30 M 4 S'::INTERVAL) - ('1 D 39 S'::INTERVAL);`
  - Yields `20:29:25`
- `SELECT('NOV-20-2020 1:23 AM EST':: TIMESTAMP WITH TIME ZONE)
- ('4D 1sec'::INTERVAL);`
  - Yields `2020-11-16 01:22:59-05`

## 135. NULL Constraint
- When data is missing
- When a table is created, use NOT NULL:
```
CREATE TABLE products(
  id SERIAL PRIMARY KEY,
  name VARCHAR(50),
  price INTEGER NOT NULL
);
```
- If the table already exists, use SET NOT NULL:
```
ALTER TABLE products
ALTER COLUMN price
SET NOT NULL;
```
- Renaming the name of column:
```
ALTER TABLE products
RENAME COLUMN deparment TO department;
```

## 136. Applying NULL constraint
- Converting NULL value into a finite number (=9999)
```
UPDATE products
SET price = 9999
WHERE price IS NULL;
```
  - `WHERE price = NULL` will not work

## 137. Default column value
- When a table is created, use DEFAULT:
```
CREATE TABLE products(
  id SERIAL PRIMARY KEY,
  name VARCHAR(50),
  price INTEGER DEFAULT 999
);
```
- If the table already exists, use SET DEFAULT:
```
ALTER TABLE products
ALTER COLUMN price
SET DEFAULT 999;
```

## 138. Unique constraint
- Avoid duplicated column values
- When a table is created, use UNIQUE:
```
CREATE TABLE products(
  id SERIAL PRIMARY KEY,
  name VARCHAR(50) UNIQUE,
  price INTEGER
);
```
- If the table already exists, use ADD UNIQUE():
```
ALTER TABLE products
ADD UNIQUE(name);
```

## 139. Applying unique constraints to multiple columns
- How to remove an existing constraint
  - Find the name of constraint from the GUI of pgadmin3
```
ALTER TABLE products
DROP CONSTRAINT products_name_key;
```
- Applying unique constraints to multiple columns
```
ALTER TABLE products
ADD UNIQUE(name, department);
```
  - Partial matching is allowed. Only when all of them match, an error is produced

## 140. Using CHECK
- Avoiding negative number
  - When a table is created, use CHECK():
```
CREATE TABLE products(
  id SERIAL PRIMARY KEY,
  name VARCHAR(50),
  price INTEGER CHECK (price > 0)
);
```
  - If the table already exists, use ADD CHECK():
```
ALTER TABLE products
ADD CHECK(price > 0);
```
- When only selected words are allowed: `mycolor VARCHAR(20) CHECK (mycolor in ('red','green','red'))`

## 141. Order constraints
- Ex) Delivery date must be after creation date
```
CREATE TABLE orders (
id SERIAL PRIMARY KEY,
name VARCHAR(50) NOT NULL,
created_at TIMESTAMP NOT NULL,
est_delivery TIMESTAMP NOT NULL,
CHECK (created_at < est_delivery)
);
```

## 143. A complicated schema
- Schema designer: drawsql.app, dbdiagram.io, sqldbm.com, ...

## 149. A sample bad schema
- How to keep the track of WHO liked WHAT?

## 150. Designing a LIKE system
- 3 TABLES of users, posts, and likes
  - Add a unique constraint with UNIQUE(user_id, post_id) in the likes TABLE

## 151. Modification for reaction system
- Adding one more column of type such as like, love, care, funny, sad, ...

## 152. Enabling LIKE and COMMENTS simultaneously
- How to make a foreign key for comment and post
  - One column for row id and another column for comment or post ?  
- Polymorphic association
  - Foreign key to refer rows in any of several related tables
  - May refer different tables by the value of the multiple columns
    - Foreign keys must point rows in the assigned table, not multiple tables
    - This guarantees data consistency - when a row in the corresponding able is deleted, the reference foreign key is updated
  - This will violate the rule of relational database design
  - Not recommended but might be used in Ruby on Rails
  - May use SUPERTABLE approach instead

## 153. Alternative of polymorphic association
- Provides separate columns for each different referred table
  - A column for comment_id and another column for post_id
- `SELECT COALESCE((4)::BOOLEAN::INTEGER,0);`
- `(4)::BOOLEAN::INTEGER,0`
  - This will convert number 4 into BOOLEAN then INTEGER, implying 1

## 154. The simplest approach
- Provides separate tables for comment history/post history
- May use UNION

## 161. Designing of a hashtag system
- Hashtags are used in posts, comments, and users (In instagram)
- Hashtag query or search is limited for posts only
  - We may not build tables of hashtag_comments nor - hashtag_users

## 162. Tables for hash tags
- Hashtag table

|id  |title | post_id|
|----|------|--------|
|1   | bird |  3     |
|2   | fun  |  5     |
|3   | fun  |  2     |
|4   | bird |  7     |

- This could be waste of memory
  - Repetition of same title
  - Title is string data

- Provide separate tables for hashtags and the link

|id  |title |
|----|------|
|1   | bird |
|2   | fun  |

|id  | hashtag_id | post_id|
|----|------------|--------|
|1   | 1          |  3     |
|2   | 2          |  5     |
|3   | 2          |  2     |
|4   | 1          |  7     |

- Repetition is managed through integer index

## 165. Number of followers or posts
- We don't build separate tables for those data
- They are `derived data`, and can be queried using existing tables

## 168. Exercising a DB for instagram
- Creating a table for users
- NOT NULL vs DEFAULT
  - NOT NULL : a value must be provided (an empty string is an input)
  - DEFAULT : When no value is INSERTED, default value is inserted
  - NOT NULL + DEFAULT : when a value is required but optional
- When one of phone or email item is required
  - `CHECK (COALESCE(phone, email) IS NOT NULL)`
  - At least, one of phone or email should not be NULL
```
CREATE  TABLE users (
	id SERIAL PRIMARY KEY,
	created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
	updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
	username VARCHAR(30) NOT NULL,
	bio      VARCHAR(400) DEFAULT '',
	avatar   VARCHAR(200),
	phone    VARCHAR(25),
	email    VARCHAR(40),
	password VARCHAR(50),
	status   VARCHAR(15)		
	CHECK(COALESCE(phone, email) IS NOT NULL)
);
```

## 169. Table of posts
- Make user_id as a foreign key column
  - Make sure the post is deleted when the user_id is removed
  - Use `ON DELETE CASCADE`
```
CREATE  TABLE posts (
	id SERIAL PRIMARY KEY,
	created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
	updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
	url      VARCHAR(200) NOT NULL,
	caption  VARCHAR(240),
	lat REAL CHECK(lat IS NULL OR (lat >= -90  AND lat <= 90 )),
	lng REAL CHECK(lng IS NULL OR (lng >= -180 AND lng <= 180)),
	user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE
);
```

## 170. Table of comments
```
CREATE  TABLE comments (
	id SERIAL PRIMARY KEY,
	created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
	updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
	contents VARCHAR(240) NOT NULL,
	user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
	post_id INTEGER NOT NULL REFERENCES posts(id) ON DELETE CASCADE
);
```

## 171. Table of likes
- How to enforce if post_id or comment_id is valid
```
CHECK (
		COALESCE((post_id)::BOOLEAN::INTEGER, 0) +
		COALESCE((comment_id)::BOOLEAN::INTEGER,0) = 1
)
```
 - COALESCE(null,0) produces 0
```
CREATE  TABLE likes (
	id SERIAL PRIMARY KEY,
	created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
	user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
	post_id    INTEGER REFERENCES posts(id) ON DELETE CASCADE,
	comment_id INTEGER REFERENCES comments(id) ON DELETE CASCADE,
	CHECK (
		COALESCE((post_id)::BOOLEAN::INTEGER, 0) +
		COALESCE((comment_id)::BOOLEAN::INTEGER,0) = 1
	)

);
```

## 172. Table of photos and captions
- Enforces a photo_tag is made by a unique combination of user and post
  - UNIQUE(user_id, post_id)
```
CREATE  TABLE photo_tags (
	id SERIAL PRIMARY KEY,
	created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
	updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
	user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
	post_id INTEGER NOT NULL REFERENCES posts(id) ON DELETE CASCADE,
	x INTEGER NOT NULL,
	y INTEGER NOT NULL,
	UNIQUE(user_id, post_id)				
);
CREATE  TABLE caption_tags (
	id SERIAL PRIMARY KEY,
	created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
	user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
	post_id INTEGER NOT NULL REFERENCES posts(id) ON DELETE CASCADE,
	UNIQUE(user_id, post_id)				
);
```

## 173. Tables of hashtags
```
CREATE  TABLE hashtags (
	id SERIAL PRIMARY KEY,
	created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
	title VARCHAR(20) NOT NULL UNIQUE
);
CREATE  TABLE hashtags_posts (
	id SERIAL PRIMARY KEY,
	hashtag_id INTEGER NOT NULL REFERENCES hashtags(id) ON DELETE CASCADE,
	post_id    INTEGER NOT NULL REFERENCES posts(id)    ON DELETE CASCADE,
	UNIQUE(hashtag_id, post_id)
);
CREATE  TABLE followers (
	id SERIAL PRIMARY KEY,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
	leader_id   INTEGER NOT NULL REFERENCES users(id)    ON DELETE CASCADE,
	follower_id INTEGER NOT NULL REFERENCES users(id)    ON DELETE CASCADE,
	UNIQUE (leader_id, follower_id)
);
```

## 175. Loading external SQL file
- In pdamdin3, right mouse button on instagram database
  - Restore -> select ig.sql
  - In the Restore option, enable `Type Of Objects - Only data`, `Dont'save - Owner`, `Queries - Single transaction`, `Disable - Trigger`
  - /usr/bin/pg_restore --host localhost --port 5432 --username "myuser" --dbname "instagram" --no-password  --data-only --disable-triggers --single-transaction --verbose "/home/hpjeon/Downloads/ig.sql"
- Error message: Unsupported version (1.14) in file header
  - Install newer version of psql and pgadmin4
- Using pgadmin4
  - Took 10.35 sec
- Removing NOT NULL constraint from the produced table
```
ALTER TABLE users
ALTER COLUMN bio
DROP NOT NULL;
```

## 178-182. Some practices
- Find max 3 IDs from user table
```
SELECT *
FROM users
ORDER BY ID DESC
LIMIT 3;
```
- Find username and captions of all posts by user ID=200
```
SELECT username, caption, posts.user_id, users.id
FROM posts
JOIN users ON posts.user_id = users.id
WHERE users.id = 200
LIMIT 10;
```
  - What if `JOIN users ON posts.user_id=200` without WHERE constraint?
    - What is missing?
- Show each username and the number of likes made by them
```
SELECT users.username, COUNT(*)
FROM likes
JOIN users ON users.id = likes.user_id
GROUP BY users.username
LIMIT 10;
```

## 183. Performance of Postgres
- How data are stored
- How index is handled
- How query is analyzed

## 184. How PosgreSQL stores the data in the disk
- From SQL `SHOW data_directory;`
- Find folders at  /var/lib/postgresql/10/main/base
```
SELECT oid, datname
FROM pg_database;
```
  - This command shows the id of each database
  - Find raw data inside of the folder from matching id
```
SELECT * FROM pg_class;
```  
  - This command shows the mapping of a table to an actual actual file (heap)

## 185. Heaps, blocks and tuples
- A heap is composed of many blocks or pages
  - Each block is 8kbytes
- A block or page is composed of many tuples or items
- A tuple or an item or a row holds all the information for on record stored in a table

## 186. Block data layout

## 187. Heap file layout
- postgresql.org/docs/current/storage-page-layout.html
- Header layout: pd_lsn, pd_checksum, pd_flags, pd_lower, pd_upper, pd_special, pd_pagesize_version, pd_prune_xid
- Item ID, size of item
- Actual data

## 188. Full table scan - how a query scan a table
- For a query like `SELECT * FROM users WHERE username = 'Riann';`
- SQL will read entire data into the memory to search
  - Will search every line
- Full Table Scan
  - From heap file to memory
  - Very expensive

## 189. How to avoid Full Table Scan
- Book-keeping the location of 'Riann'
  - **Let's make an Index**
  - Tells which block/row is for 'Riann'
  - Avoids to load entire data

## 190. How an index works
- When query is searched, extracts username and block location/row only
  - Not entire data
- Then sorts by some order
- Builds TREE data structures (B-Tree, Hash, GiST, ...)
  - Adds helpers to the root node
    - Provides a criterion which branch might be taken
  - Visits the corresponding leaf node, finding the block record

## 191. Creating an index
- `CREATE INDEX ON users(username);`
- In pdadmin4, Tables->users->indexes is found
- To delete index, `DROP INDEX users_username_idx;`

## 192. Benchmarking queries
- Use `EXPLAIN ANALYZE`
```
EXPLAIN ANALYZE SELECT *
FROM users
WHERE username = 'Emil30';
```
  - Prints execution time - around 0.04ms
- After DROP INDEX, it took 0.96ms

## 193. Downsides of indices
- Size of tables
```
SELECT pg_size_pretty(pg_relation_size('users'));
SELECT pg_size_pretty(pg_relation_size('users_username_idx'));
```
- `users` table consumes 880 kb while `users_username_idx` uses 184 kb
- Downsides of indexing
  - Can consume large disk space
  - Will slow down insert/update/delete as the index has to be updated
  - Index might not be used

## 194. Types of index
- B-Tree : General purpose
- Hash : speeds up simple equality checks
- GiST : Geometry, full-text search
  - ~~`CREATE INDEX username ON users USING GIST`~~
- SP-GiST : clustered data like dates
- GIN : For columns containing arrays or JSON
  - ~~`CREATE INDEX name on table USING GIN`~~
- BRIN : specialized for very large datasets
- Using them is not addressed in this class

## 195. Automatically generated indices
- PostgreSQL automatically creates indices for primary key and columns of unique constraints
  - They are NOT shown in Tables -> table_name -> indexes in pgadmin4
- For index table, `relkind = 'i'`
- `pg_class` shows all of objects in PostgreSQL
```
SELECT relname, relkind
FROM pg_class
WHERE relkind = 'i';
```
  - This command shows all objects of indexes
  - `users_pkey` is the index of primary key in users table, and automatically generated

## 196. Behind the scenes of indexes
```
CREATE EXTENSION pageinspect;
```
- Allows the access to the contents of the database at low level
```
SELECT *
FROM bt_metap('users_username_idx');
```
- root is found as 3
```
SELECT *
FROM bt_page_items('users_username_idx',3);
```
- Shows the B-TREE structures
```
SHOW data_directory;
```
- Shows the path where actual data are stored

## 197. Good query vs bad query using index
- SQL query -> parser -> rewrite -> planner -> execute
- Planner
  - Decides which data to be searched or fetched
  - Finds the most efficient strategy

## 198. EXPLAIN and EXPLAIN ANALYZE
- In planner, they are for benchmarking + evaluating queries, not for user in real data fetching
  - Find the most efficient strategy
- `EXPLAIN`: builds a query plan and display info about it
  - Just explain. No execution
- `EXPLAIN ANALYZE` : builds a query plan, run it, and info about it
  - In addition to the explain, run the script and shows the execution time
- pgadmin has a menu button for EXPLAIN and EXPLAIN ANALYZE
  - Profiling options can be added
![Snapshot of pgadmin](./explain_analzye.png)
- Sample results of EXPLAIN ANALYZE
```
EXPLAIN ANALYZE SELECT username, contents
FROM users
JOIN comments ON comments.user_id = users.id
WHERE username = 'Alyson14';
#####
Hash Join  (cost=8.31..1756.11 rows=11 width=81) (actual time=0.150..11.893 rows=7 loops=1)
  Hash Cond: (comments.user_id = users.id)
  ->  Seq Scan on comments  (cost=0.00..1589.10 rows=60410 width=72) (actual time=0.007..5.640 rows=60410 loops=1)
  ->  Hash  (cost=8.30..8.30 rows=1 width=17) (actual time=0.040..0.040 rows=1 loops=1)
        Buckets: 1024  Batches: 1  Memory Usage: 9kB
        ->  Index Scan using users_username_idx on users  (cost=0.28..8.30 rows=1 width=17) (actual time=0.029..0.037 rows=1 loops=1)
              Index Cond: ((username)::text = 'Alyson14'::text)
```
  - Index Scan -> Hash -> Hash Join <- Seq Scan
  - For `(cost=8.31..1756.11 rows=11 width=81)`
    - Computing cost might be 17561.11 (8.31 is the startup time) 11 rows might be produced. 81 bytes per each row might be consumed
  - For `(actual time=0.150..11.893 rows=7 loops=1)`
    - Actual computing cost is 11.893 (0.150 is the startup time). 7 rows were produced
- Following query shows the size of each column in users table
```
SELECT *
FROM pg_stats
WHERE tablename = 'users';
```
![Snapshot of pgadmin](./pg_stats.png)

## 200-202. Understanding Cost
- How the planner decides which strategy is the most efficient?
  - Cost estimation
  - postgresql.org/docs/current/runtime-config-query.html
    - random_page_cost, seq_page_cost, cpu_tuple_cost, cpu_index_tuple_cost, cpu_operator_cost, ...
    - seq_page_cost is defined as 1.0 as the basic reference
    - Users may customize those values to tune the performance
- Typical cost = (# pages read sequentially) * seq_page_cost + (# pages read a random) * random_page_cost + (# rows scanned) * cpu_tuple_cost + (# index entries scanned) * cpu_index_tuple_cost + (# times function/operator evaluated) * cpu_operator_cost
  - Or simplified version: Cost = (# pages read sequentially) * seq_page_cost + (# rows scanned) * cpu_tuple_cost
  - Actual cost estimation is more complicated

## 205. Sample exercise
- Create index for likes table
```
CREATE INDEX likes_created_at_idx ON likes(created_at);
```
- Following query uses index
```
EXPLAIN SELECT *
FROM likes
WHERE created_at < '2013-01-01';
```
- Following query just uses sequential scan, not using index
```
EXPLAIN SELECT *
FROM likes
WHERE created_at > '2013-01-01';
```
  - Why ?
  - Random access through index becomes very expensive for very large data
  - Cost was estimated and compared

## 206-207. Sample UNION query
- Show the username of users who were tagged in a caption or photo before Jan 07 2010. Also show the date they were tagged
```
EXPLAIN ANALYZE SELECT users.username, tags.created_at
FROM users
JOIN (
SELECT user_id, created_at FROM caption_tags
UNION ALL
SELECT user_id, created_at FROM photo_tags
) AS tags ON tags.user_id = users.id
WHERE tags.created_at < '2010-01-07';
```

## 208. Common Table Expression (CTE)
- Intermediate query results that can be used other main query statement
```
WITH tags AS (
	SELECT user_id, created_at FROM caption_tags
	UNION ALL
	SELECT user_id, created_at FROM photo_tags
)
SELECT users.username, tags.created_at
FROM users
JOIN tags AS tags ON tags.user_id = users.id
WHERE tags.created_at < '2010-01-07';
```
- Define with a `WITH` before the main query
  - Simple form is used to make a query easy to read
  - Recursive form is used to express complex query
- Simple form of CTE doesn't change the cost or planner

## 209-210. Recursive CTE
- Useful for tree or graph-style data structure
- Must use a UNION keyword
- Use existing templates or examples - very difficult to understand
```
WITH RECURSIVE countdown(val) AS (
	SELECT 3 AS val
	UNION
	SELECT val - 1 FROM countdown WHERE val > 1
)
SELECT *
FROM countdown;
```
  - Produces a column of 3, 2, 1

## 212-213. A sample recursive CTE
```
WITH RECURSIVE suggestions(leader_id, follower_id, depth) AS (
	SELECT leader_id, follower_id, 1 AS depth
	FROM followers
	WHERE follower_id=1000
	UNION
	SELECT followers.leader_id, followers.follower_id, depth+1
	FROM followers
	JOIN suggestions ON suggestions.leader_id = followers.follower_id
	WHERE depth <3
)
SELECT DISTINCT users.id, users.username
FROM suggestions
JOIN users ON users.id = suggestions.leader_id
WHERE depth > 1
LIMIT 30;
```

## 214. A same query
- Show the most popular users - the users who were tagged the most (caption_tags, photo_tags)
```
SELECT username, COUNT(*)
FROM users
JOIN(
	SELECT user_id FROM caption_tags
	UNION ALL
	SELECT user_id FROM photo_tags
) AS merge ON merge.user_id = users.id
GROUP BY users.username
ORDER BY COUNT(*) DESC
LIMIT 10;
```
- We may not want to use UNION many times for merge tables. How to merge existing tables?

## 215. Merging tables
- We do not want to delete existing tables
- Let's create a VIEW

## 216. Creating a View
```
CREATE VIEW tags AS (
	SELECT id, created_at, user_id, post_id, 'photo_tag' AS type FROM photo_tags
	UNION ALL
	SELECT id, created_at, user_id, post_id, 'caption_tag' AS type FROM caption_tags
);
SELECT username, COUNT(*)
FROM users
JOIN tags ON tags.user_id = users.id
GROUP BY users.username
ORDER BY COUNT(*) DESC
LIMIT 10;
```

## 217. A sample VIEW
- A VIEW for 10 the most recent post
```
CREATE VIEW recent_post AS (
	SELECT *
	FROM posts
	ORDER BY created_at DESC
	LIMIT 10
);
```

## 218. Deleting and changing VIEWs
- CREATE OR REPLACE
```
CREATE OR REPLACE VIEW recent_post AS (
	SELECT *
	FROM posts
	ORDER BY created_at DESC
	LIMIT 15
);
```
- DROP for deleting views
```
DROP VIEW recent_post;
```

## 219. Materialized VIEWs
- VIEW is a kind of macro, and is executed every time when a query sees the VIEW
  - Ref: https://dba.stackexchange.com/questions/7242/does-sql-server-calculate-views-every-time-when-someone-query-the-views
- Materialized VIEW: executed only at specific times, and can be referenced

## 220-223. A sample query using LEFT JOIN
- For each week, show the number of likes that posts and comments received. Use created_at of the post and comment tables, not when the like was received
```
SELECT
	date_trunc('week', COALESCE(posts.created_at, comments.created_at)) AS week,
	COUNT(posts.id) AS num_likes_for_posts,
	COUNT(comments.id) AS num_likes_for_comments
FROM likes
LEFT JOIN posts ON posts.id = likes.post_id
LEFT JOIN comments ON comments.id = likes.comment_id
GROUP BY week
ORDER BY week;
```
- This takes 1.7 sec
```
CREATE MATERIALIZED VIEW weekly_likes AS (
	SELECT
		date_trunc('week', COALESCE(posts.created_at, comments.created_at)) AS week,
		COUNT(posts.id) AS num_likes_for_posts,
		COUNT(comments.id) AS num_likes_for_comments
	FROM likes
	LEFT JOIN posts ON posts.id = likes.post_id
	LEFT JOIN comments ON comments.id = likes.comment_id
	GROUP BY week
	ORDER BY week
) WITH DATA;
```
- Now `SELECT * FROM weekly_likes;` takes 0.1msec
- When data is modified, weekly_likes VIEW doesn't update automatically
- `REFRESH MATERIALIZED VIEW weekly_likes;` will update the materialized VIEW

## 224. Transactions
- How to make sure the entire multiple operations are done completely or when broken, how to roll back
- BEGIN, COMMIT, ROLLBACK

## 225-226. A sample transaction
```
CREATE TABLE accounts(
id SERIAL PRIMARY KEY,
name VARCHAR(20) NOT NULL,
balance INTEGER NOT NULL
);
INSERT INTO accounts (name, balance)
VALUES
('Gia',100),
('Alyson',100);
```
- `BEGIN;` starts a transaction
- `COMMIT`: merges changes to the main data pool
- `ROLLBACK`: cancels the change
- Bad commands will put the transaction in an aborted state. ROLLBACK must be executed
- With connection 1)
```
BEGIN;
UPDATE accounts
SET balance = balance - 50
WHERE name = 'Alyson';
UPDATE accounts
SET balance = balance + 50
WHERE name = 'Gia';
COMMIT;
```
- With connection 2)
  - Balance change is not found until COMMIT is made from connection 1)

## 228. Manual ROLLBACK
```
BEGIN;
SELECT * FROM nonexisting_table;
```
- SQL will crash. Any new query will not be executed
```
ROLLBACK;
```
- After ROLLBACK, an appropriate query will work

## 229-230. Schema migration
- Request: column name of 'contents' as 'body'
- `ALTER TABLE comments RENAME COLUMN contents TO body;`
  - Will break existing API or queries
- Changes of DB and API must be made in the same time
- Or Need to tie the structure of the database into the code
  - Not from PGAdmin
  - Needs schema Migration File: C, Python, Java, ...
- Migration File
  - Up: Apply or change the DB structure
  - Down: Undo Up or revert
- Ex) 1
  - Up: CREATE TABLE comments ( id SERIAL PRIMARY KEY, created_at ....);
  - Down: DROP TABLE comments;
- Ex) 2
  - Up: ALTER TABLE comments RENAME COLUMN contents TO body;
  - Down: ALTER TABLE comments
   RENAME COLUMN body TO contents;

## 232. Migration libraries
- Python: alembic, yoyo-migration
- Java: flywaydb
- Javascript: node-pg-migrate, sequelize, db-migrate
- Recommendation: instead of automatic SQL query from the library, write your own SQL manually

## 234-237. A sample node js project
- `mkdir ig`
- `cd ig`
- `npm init -y`
- `npm install node-pg-migrate pg`
- Edit package.json
```
{
  "name": "ig",
  "version": "1.0.0",
  "description": "",
  "main": "index.js",
  "scripts": {
    "migrate": "node-pg-migrate"
  },
  "keywords": [],
  "author": "",
  "license": "ISC",
  "dependencies": {
    "node-pg-migrate": "^5.9.0",
    "pg": "^8.5.1"
  }
}
```
- `npm run migrate create table comments`
- Edit migrations/*_table-comments.js
```
/* eslint-disable camelcase */
exports.shorthands = undefined;
exports.up = pgm => {
  pgm.sql(`
    CREATE TABLE comments(
      id SERIAL PRIMARY KEY,
      created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
      updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
      contents VARCHAR(240) NOT NULL
    );
    `);
};
exports.down = pgm => {
  pgm.sql(`
    DROP TABLE comments;
    `)
};
```
- `DATABASE_URL=postgres://postgres:__passwd__@localhost:5432/socialnetwork npm run migrate up`
  - pgadmin can find the table comments is produced
- `DATABASE_URL=postgres://postgres:__passwd__@localhost:5432/socialnetwork npm run migrate down`
  - Now the table comments is gone
- `npm run migrate create rename contents to body`
- Edit the newly produced json
```
/* eslint-disable camelcase */
exports.shorthands = undefined;
exports.up = pgm => {
  pgm.sql(`
    ALTER TABLE comments
    RENAME COLUMN contents TO body;
    `)
};
exports.down = pgm => {
  pgm.sql(`
  ALTER TABLE comments
  RENAME COLUMN body TO contents;
  `)
};
```
- `DATABASE_URL=postgres://postgres:__passwd__@localhost:5432/socialnetwork npm run migrate up`
  - As there are two json files, 1) it will create a table and 2) will ALTER the table
```
### MIGRATION 1607647512791_table-comments (UP) ###
    CREATE TABLE comments(
      id SERIAL PRIMARY KEY,
      created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
      updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
      contents VARCHAR(240) NOT NULL
    );
    ;
INSERT INTO "public"."pgmigrations" (name, run_on) VALUES ('1607647512791_table-comments', NOW());
### MIGRATION 1607648312077_rename-contents-to-body (UP) ###
   ALTER TABLE comments
    RENAME COLUMN contents TO body;
    ;
INSERT INTO "public"."pgmigrations" (name, run_on) VALUES ('1607648312077_rename-contents-to-body', NOW());
```
- `DATABASE_URL=postgres://postgres:__passwd__@localhost:5432/socialnetwork npm run migrate down`
  - Will revert those steps
```
### MIGRATION 1607648312077_rename-contents-to-body (DOWN) ###
  ALTER TABLE comments
  RENAME COLUMN body TO contents;
  ;
DELETE FROM "public"."pgmigrations" WHERE name='1607648312077_rename-contents-to-body';
Migrations complete!
```

## 238-239. Schema and data migrations
- Moving latitude and longitudinal columns into a single column of location
  - Schema migration: adding a new column/removing old columns
  - Data migration: merging two data into a single data type
    - What if there are millions of data?
  - Running  migration in a transaction might be recommended
    - Newly added rows might not be processed from such a transaction: potential data loss
- May split schema migration and data migration

## 240. Proper migrations of schema and data
- Add column loc
- Let API fill data of lat/lng/loc
- Copy lat/lng to loc
- Update API to write loc only
- Drop columns of lat/lng
- Can avoid any data loss

## 241. npm command to create tables
- `npm run migrate create add posts table`
- Edit js file at migration folder
```
/* eslint-disable camelcase */
exports.shorthands = undefined;
exports.up = pgm => {
  pgm.sql(`
    CREATE TABLE posts (
      id SERIAL PRIMARY KEY,
      url VARCHAR(300),
      lat NUMERIC,
      lng NUMERIC
    );
    `);
};
exports.down = pgm => {
  pg.sql(`
    DROP TABLE posts;
    `);
};
```

## 242. Web server using express
- `npm install express pg`
- Make index.js
```
const express = require('express');
const pg = require('pg');
const pool = new pg.Pool({
  host: 'localhost',
  port: 5432,
  database: 'socialnetwork',
  user: 'postgres',
  password: 'XXXXXX'
  });
//pool.query('SELECT 1+1;').then((res) => console.log(res));
const app = express();
app.use(express.urlencoded({ extended:true }));

app.get('/posts', async(req,res) => { const {rows } = await pool.query(`
  SELECT * FROM posts;
  `);
  res.send(`
    <table>
    <thead>
    <tr>
    <th>id</th>
    <th>lng</th>
    <th>lat</th>
    </tr>
    </thead>
    <tbody>
    ${rows.map(row => {
      return `
        <tr>
        <td>${row.id} </td>
        <td>${row.lng} </td>
        <td>${row.lat} </td>
        </tr>
      `;
    }). join('')}
    </tbody>
    </table>
    <form method="POST">
     <h3> Create Post </h3>
     <div>
        <label> Lng</label>
        <input name = "lng"/>
      </div>
      <div>
        <label> Lat</label>
        <input name = "lat"/>
      </div>
      <button type = "submit">Create</button>
      </form>
    `);
});
app.post('/posts', async(req,res) => {
  const { lng, lat } = req.body;
  await pool.query(
    'INSERT INTO posts (lat, lng) VALUES ($1, $2);',
    [lat, lng]
  );
  res.redirect('/posts')
});
app.listen(3005, () => {
  console.log('Listening on port 3005');
})
```
- `node index.js` will start web server
- Open http://localhost:3005/posts in the web browser
- `npm run migrate create add loc to posts`
- Edit migration file
```
/* eslint-disable camelcase */
exports.shorthands = undefined;
exports.up = pgm => {
  pgm.sql(`
    ALTER TABLE posts
    ADD COLUMN loc POINT;
    `);
};
exports.down = pgm => {
  pgm.sql(`
    ALTER TABLE posts
    DROP COLUMN loc;
    `);
};
```
- `DATABASE_URL=postgres://postgres:__passwd__@localhost:5432/socialnetwork npm run migrate up`
  - Now loc column is added

## 245. Writing data to the existing/new columns
- Modify:
```
app.post('/posts', async(req,res) => {
  const { lng, lat } = req.body;
  await pool.query(
    'INSERT INTO posts (lat, lng) VALUES ($1, $2);',
    [lat, lng]
  );
  res.redirect('/posts')
});
```
- To:
```
app.post('/posts', async(req,res) => {
  const { lng, lat } = req.body;
  await pool.query(
    'INSERT INTO posts (lat, lng, loc) VALUES ($1, $2, $3);',
    [lat, lng, `(${lng}, ${lat})`]
  );
  res.redirect('/posts')
});
```
- Restart index.js and add new data. Now loc column is updated

## 246. Copying data to the loc column
- DO NOT update at once - may overload the server
- Split data and run batch jobs many times, updating data piece by piece
- Transaction Locks
  - During the transaction, handled data are locked
  - Any transaction accessing the locked data will be queued
  - Pending queue will be executed the locked transaction is committed or rolled-back

## 247. Updating values of loc column
- mkdir -p migrations/data
- Edit 01-lng-lat-to-loc.js
```
const pg = require('pg');
const pool = new pg.Pool({
  host: 'localhost',
  port: 5432,
  database: 'socialnetwork',
  user: 'postgres',
  password: 'XXXXXXXXXXX'
  });
pool.query(`
  UPDATE posts
  SET loc = POINT(lng, lat)
  WHERE loc IS NULL;
`)
.then(() =>{
  console.log('Update Complete');
  pool.end();
})
.catch((err) => console.error(error.message));
```
- `node 01-lng-lat-to-loc.js` will update the server

## 251. New project using Node API
- Javascript API to interact with socialnetwork table

## 252. Initial setup
- `mkdir api; cd api`
- `npm install dedent express jest node-pg-migrate nodemon pg pg-format supertest node-pg-migrate`

## 253. One fast migration
- `npm init -y`
- Edit package.json
```
{
  "name": "social-repo",
  "version": "1.0.0",
  "description": "",
  "main": "index.js",
  "scripts": {
    "migrate": "node-pg-migrate",
    "start": "nodemon index.js"
  },
  "keywords": [],
  "author": "",
  "license": "ISC"
}
```
 - `npm run migrate create add users table`
 - Now edit migrations/XXX-add-users-table.js
 ```
 /* eslint-disable camelcase */
 exports.shorthands = undefined;
 exports.up = pgm => {
   pgm.sql(`
     CREATE TABLE users(
       id SERIAL PRIMARY KEY,
       created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
       updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
       bio VARCHAR(400),
       username VARCHAR(30) NOT NULL
     );`);
 };
 exports.down = pgm => {
   pgm.sql(`
     DROP TABLE users;
     `);
 };
```
- Drop socialnetwork database from pgadmin4 and create it again
  - If a transaction is not handled appropriately, the pg_locks is applied and the db cannot be dropped.
  - Couldn't drop a database from pgadmin4
  - `sudo -u postgres psql; DROP DATABASE socialnetwork` has dropped the table
- `DATABASE_URL=postgres://postgres:__passwd__@localhost:5432/socialnetwork npm run migrate up`
  - pgadmin will show that users table is produced

## 254. Building the users router
- Make src/app.js
```
const express = require('express');
const usersRouter = require('./routes/users');
module.exports = () => {
  const app = express();
  app.use(express.json());
  app.use(usersRouter);
  return app;
};
```
- Make src/routes/users.js
```
const express = require('express');
const router = express.Router();
router.get('/users', async(req, res) => {});
router.get('users/:id', async(req, res) => {});
router.post('/users', async(req,res) => {});
router.post('/users/:id', async(req,res) => {});
router.delete('/users/:id', async(req,res) => {});
module.exports = router;
```

## 255. Understanding Connection Pools
- A pool internally maintains several different clients that can be re-used
- Make src/pool.js
```
const pg = require('pg');
class Pool {
  _pool = null;
  connect(options) {
    this._pool = new pg.Pool(options)
    return this._pool.query('SELECT 1+1;');
    // enforces a client is made when first connection is made
  }
}
module.exports = new Pool();
```

## 256. Validating connection credentials
- Make index.js
```
const app = require('./src/app.js');
const pool = require('./src/pool.js');
pool.connect({
  host: 'localhost',
  port: 5432,
  database: 'socialnetwork',
  user: 'postgres',
  password: 'XXXXXXXXX'
});
app().listen(3005, () =>{
  console.log('Listening on port 3005');
});
```

## 257. Query and close
- Add following functions to pool.js
```
close() {
  return this._pool.end();
}
query(sql) {
  return this._pool.query(sql); // this has security issue!!!
}
```

## 259. Creating a repository
- Make src/repos/user-repo.js
```
const pool = require('../pool');
class UserRepo {
  static async find() {
    const result = await pool.query('SELECT * FROM users;');
    return result.rows;
  }
  static async findById() {  }
  static async insert() {  }
  static async update() {  }
  static async delete() {  }
}
module.exports = UserRepo
```
- Update users.js accordingly
```
const express = require('express');
const UserRepo = require('../repos/user-repo');
const router = express.Router();
router.get('/users', async(req, res) => {
  const users = await UsersRepo.find();
  res.send(users);
});
router.get('users/:id', async(req, res) => {});
router.post('/users', async(req,res) => {});
router.post('/users/:id', async(req,res) => {});
router.delete('/users/:id', async(req,res) => {});
module.exports = router;
```

## 260
- `npm install nodemon`
- `npm run start`
- postman or REST client of MS code
- camel casing of Javascript vs different style: created_at vs createdAt

## 262
- Converted created_at key into createdAt

## 264. SQL Injection Exploits
- `SELECT * FROM users WHERE id= ${id};` from API
- If the request sends `http://localhost:3005/users/1;DROP TABLE users;`, the table will be dropped
  - `${id}` is handled a string

## 265. Prepared statements for SQL Injection
- How to sanitize user-provided values
  - Provide appropriate API
  - Let Postgres sanitize
- PREPARE statement may map the data type for arguments

## 266. Preventing SQL Injection
- From: `SELECT * FROM users WHERE id= ${id};`
- To: `SELECT * FROM users WHERE id= $1` using API. Deliver arguments using appropriate mapping
  - Let only integer data type be mapped, not the entire string

## 274. Adding disconnection after tests
