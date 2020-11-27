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

##
