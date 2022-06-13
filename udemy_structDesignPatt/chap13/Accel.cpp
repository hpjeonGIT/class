#include "Accel.h"
double Accelerometer::GetHorizontalAxis() {
  std::uniform_int_distribution<> dist{-10,10};
  return dist(m_Engine);
}
double Accelerometer::GetVerticalAxis() {
  std::uniform_int_distribution<> dist{-10,10};
  return dist(m_Engine);
}