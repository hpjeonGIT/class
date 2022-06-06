#include "anim.h"
#include <iostream>
#include <thread>
#include <cstdlib>
using namespace std::literals::chrono_literals;
Animation::Animation(std::string_view animFile) {
  std::cout << "[Animation] Loading " << animFile << ' ';
  for (int i=0;i<10;++i) {
    std::cout << ".";
    std::this_thread::sleep_for(200ms);
  }
  std::cout << '\n';
  m_AnimationData.assign("^^^^^");
}
Vehicle::Vehicle(int mSpeed, int mHitPoints, const std::string& mName,
          std::string_view animFile, const Position& mPosition) 
          : m_Speed{mSpeed}, m_HitPoints{mHitPoints},
            m_Name{mName}, m_Position{mPosition} {
            m_pAnimation = new Animation{animFile};
}
Vehicle::~Vehicle() { 
  delete m_pAnimation;
}
void Vehicle::SetAnimationData(const std::string& animData) { 
  m_pAnimation->SetAnimationData(animData);
}
const std::string& Vehicle::GetAnimation() const {
  return m_pAnimation->GetAnimationData();
}
Vehicle::Vehicle(const Vehicle &other):  // copy constructor
  m_Speed{other.m_Speed}, m_Name{other.m_Name}, 
  m_HitPoints{other.m_HitPoints}, m_Position{other.m_Position} {
    m_pAnimation = new Animation();
    m_pAnimation->SetAnimationData(other.GetAnimation());
}
Vehicle & Vehicle::operator=(const Vehicle &other) {  // assign operator
  if (this != &other) {
    m_Speed = other.m_Speed;
    m_Name = other.m_Name;
    m_HitPoints = other.m_HitPoints;
    m_Position = other.m_Position;
    m_pAnimation->SetAnimationData(other.GetAnimation());
  }
  return *this;
}
Vehicle::Vehicle(Vehicle &&other) noexcept : // Move constructor
    m_Speed{other.m_Speed}, m_Name{other.m_Name}, 
    m_HitPoints{other.m_HitPoints}, m_Position{other.m_Position} {
    m_pAnimation = other.m_pAnimation;
    other.m_pAnimation = nullptr;
    other.m_Position = {0,0};
    other.m_HitPoints = 0;
    other.m_Speed = 0;
    other.m_Name.clear();
}
Vehicle & Vehicle::operator=(Vehicle &&other) noexcept { // move operator
  if (this != &other) {
    m_Speed = other.m_Speed;
    m_Name = other.m_Name;
    m_HitPoints = other.m_HitPoints;
    m_Position = other.m_Position;
    delete m_pAnimation;
    m_pAnimation  = other.m_pAnimation;
    other.m_pAnimation = nullptr;
    other.m_Position = {0,0};
    other.m_HitPoints = 0;
    other.m_Speed = 0;
    other.m_Name.clear();
  }
  return *this;
}  
void Vehicle::Update() { }
void GreenCar::Update() {
  std::cout << "[" << GetName() << "]\n"
    << "\tAnimation:" << GetAnimation() << "\n"
    << "\tSpeed:" << GetSpeed() << "\n"
    << "\tHitPoints:" << GetHitPoints() << "\n"
    << "\tPosition:" << GetPosition() << "\n";
}
void RedCar::Update() {
  std::cout << "[" << GetName() << "]\n"
    << "\tAnimation:" << GetAnimation() << "\n";
  if (m_Dist(m_Engine)) {
    std::cout << "\tIncrease speed temporarily:" << GetSpeed() * m_SpeedFactor << "\n";
  } else {
    std::cout << "\tSpeed:" << GetSpeed() << "\n";
  }
  std::cout << "\tHitPoints:" << GetHitPoints() << "\n"
    << "\tPosition:" << GetPosition() << "\n";
}
void BlueBus::Update() {
  std::cout << "[" << GetName() << "]\n"
    << "\tAnimation:" << GetAnimation() << "\n";
  if (m_Dist(m_Engine)) {
    std::cout << "\tMoving out of the way\n";
  } 
  std::cout << "\tSpeed:" << GetSpeed() << "\n";
  std::cout << "\tHitPoints:" << GetHitPoints() << "\n"
    << "\tPosition:" << GetPosition() << "\n";
}
void YellowBus::Update() {
  std::cout << "[" << GetName() << "]\n"
    << "\tAnimation:" << GetAnimation() << "\n";
  if (m_Dist(m_Engine)) {
    std::cout << "\tMoving out of the way\n";
  } 
  std::cout << "\tSpeed:" << GetSpeed() << "\n";
  std::cout << "\tHitPoints:" << GetHitPoints() << "\n"
    << "\tPosition:" << GetPosition() << "\n";
}
Vehicle* RedCar::Clone() {
  std::cout << "Cloning-> "<< GetName() << "\n";
  return new RedCar{*this};
}
Vehicle* GreenCar::Clone() {
  std::cout << "Cloning-> "<< GetName() << "\n";
  return new GreenCar{*this};
}
Vehicle* YellowBus::Clone() {
  std::cout << "Cloning-> "<< GetName() << "\n";
  return new YellowBus{*this};
}
Vehicle* BlueBus::Clone() {
  std::cout << "Cloning-> "<< GetName() << "\n";
  return new BlueBus{*this};
}
void GameManager::Run() {
  m_Vehicles.push_back(Create("redcar",   30,10, "RedCar",    "red.anim",   {0,0}));
  m_Vehicles.push_back(Create("greencar",  30,10, "GreenCar",  "green.anim", {100,0}));
  m_Vehicles.push_back(Create("yellowbus", 30,10, "YellowBus", "rbus.anim",  {100,200}));
  m_Vehicles.push_back(Create("bluebus",   30,10, "BlueBus",   "bbus.anim",  {100,200}));
  int count{5};
  while(count !=0) {
    std::this_thread::sleep_for(1s);
    // system("cls"); // for windows
    std::system("clear");
    for (auto vehicle: m_Vehicles) {
      vehicle->Update();
    }
    if (count ==2) {
        //m_Vehicles.push_back(Create("redcar", 30,15, "RedCar", "red.anim", {0,0}));
        auto vehicle = m_Vehicles[0]->Clone();
        vehicle->SetPosition({50,50});
        vehicle->SetHitPoints(15);
        m_Vehicles.push_back(vehicle);
    }
    if (count ==3) {
      //m_Vehicles.push_back(Create("yellowbus", 20,20, "YellowBus", "rbus.anim", {0,0}));
      auto vehicle = m_Vehicles[2]->Clone();
      vehicle->SetPosition({150,150});
      vehicle->SetSpeed(10);
      m_Vehicles.push_back(vehicle);
    }
    --count;
  }
}
GameManager::~GameManager() {
  for (auto vehicle:m_Vehicles) {
    delete vehicle;
  }
}
int main() {
  GameManager mgr;
  mgr.Run();
}