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
