#pragma once
#include <random>
class Input {
public:
  virtual bool Up() = 0;
  virtual bool Down() = 0;
  virtual bool Left() = 0;
  virtual bool Right() = 0;
  virtual ~Input() = default;
};
class Keyboard: public Input {
  std::default_random_engine m_Engine{12345};
  bool SimulationInput();
public:
  bool Up() override;
  bool Down() override;
  bool Left() override;
  bool Right() override;
};
class Accelerometer {
  std::default_random_engine m_Engine{12345};  
public:
  double GetHorizontalAxis();
  double GetVerticalAxis();
};
class AccelAdapter: public Input, private Accelerometer {
  Accelerometer m_Accel;
public:
  bool Up() override;
  bool Down() override;
  bool Left() override;
  bool Right() override;
};