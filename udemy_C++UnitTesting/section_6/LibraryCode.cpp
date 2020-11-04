#include "LibraryCode.hpp"
#include <algorithm>
#include <iostream>

bool isPositive(int x)
{
  return x>=0;
}

int countPositives(std::vector<int> const& inputVector)
{
  return std::count_if(inputVector.begin(), inputVector.end(), isPositive);
}


int add(int a, int b)
{
  return a+b;
}

void toUpper(char *inputString)
{
  for (size_t i=0; i<strlen(inputString); i++)
  {
    inputString[i] = toupper(inputString[i]);
  }
  std::cout << inputString << std::endl;
}

double mySqrt(double input)
{
  if (input <0)
  {
    std::cout <<"Exception thrown" << std::endl;
    throw std::runtime_error("Negative argument!");
  }
  return sqrt(input);
}
