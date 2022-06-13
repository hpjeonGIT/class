#include "flight.h"
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
  Keyboard k;
  Game(&k);
}