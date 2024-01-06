## Title: Learn Perl 5 By Doing It
- Instructor: John Purcell

## Section 1: Basic Perl: Getting Started

1. Installing Perl and Some Great Free Editors

2. Hello World
```pl
use strict;
use warnings;

sub main {
  print "Hello World\n";
}
main()
```
- To run, `perl test.pl`

3. Downloading Text and Images - Updated
- https://github.com/caveofprogramming/learn-perl
- learn-perl/Tutorial3
- download_html.pl:
```pl
use strict;
use warnings;
use LWP::UserAgent;
use IO::Socket::SSL;

my $ua = LWP::UserAgent->new(cookie_jar=>{});

$ua->ssl_opts(
    'SSL_verify_mode' => IO::Socket::SSL::SSL_VERIFY_NONE, 
    'verify_hostname' => 0
);

my $request = new HTTP::Request('GET', 'https://caveofpython.com/');

my $response = $ua->request($request);

unless($response->is_success()) {
    die $response->status_line();
}

my $content = $response->decoded_content();

print($content);

print("Completed")
```
- download_image.pl:
```pl
use strict;
use warnings;
use LWP::UserAgent;
use IO::Socket::SSL;

my $ua = LWP::UserAgent->new(cookie_jar=>{});

$ua->ssl_opts(
    'SSL_verify_mode' => IO::Socket::SSL::SSL_VERIFY_NONE, 
    'verify_hostname' => 0
);

my $link = 'https://i0.wp.com/caveofpython.com/wp-content/uploads/2023/03/robot.png?w=996&ssl=1';

my $response = $ua->mirror($link, './robot.png');

unless($response->is_success()) {
    die $response->status_line();
}

print("\nCompleted")
```


4. Downloading Text and Images with Perl (Old version)

5. Arrays and Checking Whether Files exist
- Single quote '': as literal
- Double quote"": when special characters are used
- `-f` check if file exists or not
- Variable with `$`
- Array with `@`
  - Comma after the last item
  - For loop using array items
    - `foreach my $file (@files) {...}`
- To turn off output beffering, Use `$|=1;`
```pl
use strict;
use warnings;
# comment
$|=1;
sub main {
  my $file = './serval.png';
  my @files = ($file, './serval2.png',);
  foreach my $f0(@files) {  # @f0 will iterate over items in an array @files
    if( -f $f0) {  # -f returns true if $f0 exists
      print "Found file $f0\n";
    }
    else {
      print "File $f0 not foud\n";
    }
  }
}
main();
```

6. Reading Files and Beginning Regular Expressions
- `die;` let the program die
  - `die "some text";` prints the message
- `<INPUT>`: reads one line from INPUT
- `=~ /REGEX/`: check if REGEX expression matches or not
```pl
use strict;
use warnings;
# comment
$|=1;
sub main {
  my $file = './myfile.txt';
  open(INPUT, $file) or die("Input file $file not found\n");
  while(my $line = <INPUT>) {
    if($line =~ /ello/) {  # if any 'ello' is found in the line
      print $line;
    }
  }
  close(INPUT);
}
main();
```

7. Writing Files and Replacing Text

8. Wildcards in Regular Expressions

9. Groups: Finding Out What You Actually Matched

10. Quantifiers: Greedy vs. Non-Greedy

11. Escape Sequences

12. Numeric Quantifiers

13. Test your Perl and Regex Knowledge - First Test

## Section 2: More on Reading Files Line by Line: Tips, Tricks and Vital Knowledge

## Section 3: Web Scraping and More Regular Expressions

## Section 4: Building a Complete Program: Command Line Options

## Section 5: Parsing XML and Complex Data Structures

## Section 6: Working with Databases

## Section 7: Perl One-Liners

## Section 8: Modules and OO Perl

## Section 9: Web Application Basics

## Section 10: Basic Sysadmin Tasks

## Section 11: Conclusion

## Section 12: Appendix 1: Example Data

## Section 13: Appendix 2: Alternate Systems

## Section 14: Extras
