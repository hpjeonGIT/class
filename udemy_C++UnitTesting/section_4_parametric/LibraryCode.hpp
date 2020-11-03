#pragma once

class Validator
{
  public:
    Validator(int low, int high);
    bool inRange(int valueToTest);
  private:
    int mLow, mHigh;
};
