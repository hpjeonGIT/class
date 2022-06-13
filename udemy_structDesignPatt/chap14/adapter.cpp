#include "adapter.h"
#include <thread>
#include <iostream>
#include <chrono>
using namespace std::chrono_literals;
bool Keyboard::SimulationInput() {
  std::bernoulli_distribution dist {.5};
  return dist(m_Engine);
}
bool Keyboard::Up() { return SimulationInput(); }
bool Keyboard::Down() { return SimulationInput(); }
bool Keyboard::Left() { return SimulationInput(); }
bool Keyboard::Right() { return SimulationInput(); }
double Accelerometer::GetHorizontalAxis() {
  std::uniform_int_distribution<> dist{-10,10};
  return dist(m_Engine);
}
double Accelerometer::GetVerticalAxis() {
  std::uniform_int_distribution<> dist{-10,10};
  return dist(m_Engine);
}
bool AccelAdapter::Up() {return m_Accel.GetVerticalAxis() > 0;}
bool AccelAdapter::Down() {return m_Accel.GetVerticalAxis() < 0;}
bool AccelAdapter::Left() {return m_Accel.GetHorizontalAxis() <0;}
bool AccelAdapter::Right() {return m_Accel.GetHorizontalAxis() >0;}
void Game(Input *pInput) {
  int count{5};
  while(count != 0) {
    std::cout  << "================\n";
    if (pInput->Up()) { std::cout << "Pitch up\n"; }
    else if (pInput->Down()) { std::cout << "Pitch down\n"; }
    else { std::cout << "Plane is level\n"; }
    if (pInput->Left()) { std::cout << "Plane is turning left\n"; }
    else if (pInput->Right()) { std::cout << "Plane is turning right\n"; }
    else { std::cout << "Plane is flying straight\n"; }
    std::cout << std::endl;
    std::this_thread::sleep_for(1s);
    --count;
  }
}
int main() {
  //Keyboard k;
  AccelAdapter k;
  Game(&k);
}