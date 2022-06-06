#include "reduced.h"
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
Vehicle::Vehicle() {
  m_pAnimation = new Animation{};
}
Vehicle::Vehicle(int mSpeed, int mHitPoints, const std::string& mName,
          std::string_view animFile, const Position& mPosition, 
          const std::string& mColor) 
          : m_Speed{mSpeed}, m_HitPoints{mHitPoints},
            m_Name{mName}, m_Position{mPosition}, m_Color{mColor} {
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
  m_HitPoints{other.m_HitPoints}, m_Position{other.m_Position},
  m_Color{other.m_Color} {
    m_pAnimation = new Animation();
    m_pAnimation->SetAnimationData(other.GetAnimation());
}
Vehicle & Vehicle::operator=(const Vehicle &other) {  // assign operator
  if (this != &other) {
    m_Speed = other.m_Speed;
    m_Name = other.m_Name;
    m_HitPoints = other.m_HitPoints;
    m_Position = other.m_Position;
    m_Color = other.m_Color;
    m_pAnimation->SetAnimationData(other.GetAnimation());
  }
  return *this;
}
Vehicle::Vehicle(Vehicle &&other) noexcept : // Move constructor
    m_Speed{other.m_Speed}, m_Name{other.m_Name}, 
    m_HitPoints{other.m_HitPoints}, m_Position{other.m_Position},
    m_Color{other.m_Color} {
    m_pAnimation = other.m_pAnimation;
    other.m_pAnimation = nullptr;
    other.m_Position = {0,0};
    other.m_HitPoints = 0;
    other.m_Speed = 0;
    other.m_Name.clear();
    other.m_Color.clear();
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
void Car::Update() {
  std::cout << "[" << GetColor() << GetName() << "]\n"
    << "\tAnimation:" << GetAnimation() << "\n";
  if (m_Dist(m_Engine) && GetColor() == "Red") {
    std::cout << "\tIncrease speed temporarily:" << GetSpeed() * m_SpeedFactor << "\n";
  } else {
    std::cout << "\tSpeed:" << GetSpeed() << "\n";
  }
  std::cout << "\tHitPoints:" << GetHitPoints() << "\n"
    << "\tPosition:" << GetPosition() << "\n";
}
void Bus::Update() {
  std::cout << "[" << GetColor() << GetName() << "]\n"
    << "\tAnimation:" << GetAnimation() << "\n";
  if (m_Dist(m_Engine)) {
    std::cout << "\tMoving out of the way\n";
  } 
  std::cout << "\tSpeed:" << GetSpeed() << "\n";
  std::cout << "\tHitPoints:" << GetHitPoints() << "\n"
    << "\tPosition:" << GetPosition() << "\n";
}
Vehicle* Car::Clone() {
  std::cout << "Cloning-> "<< GetName() << "\n";
  return new Car{*this};
}
Vehicle* Bus::Clone() {
  std::cout << "Cloning-> "<< GetName() << "\n";
  return new Bus{*this};
}
GameManager::~GameManager() {
  for (auto vehicle:m_Vehicles) {
    delete vehicle;
  }
}
std::vector<std::string> VehiclePrototypes::GetKeys() {
  std::vector<std::string> keys {} ;
  keys.reserve(m_Prototypes.size());
  for(const auto &kv : m_Prototypes) {
    keys.push_back(kv.first);
  }
  return keys;
}
void VehiclePrototypes::RegisterPrototype(const std::string &key, Vehicle *prototype) {
  if (auto it = m_Prototypes.find(key); it == end(m_Prototypes)) {
    m_Prototypes[key] = prototype;
  } else {
    std::cout << "Key already exists\n";
  }
}
Vehicle * VehiclePrototypes::DeregisterPrototype(const std::string &key) {
  if (auto it= m_Prototypes.find(key); it !=end(m_Prototypes)) {
    auto vehicle = m_Prototypes[key];
    m_Prototypes.erase(key);
    return vehicle;
  }
  return nullptr;
}
Vehicle * VehiclePrototypes::GetPrototype(const std::string &key) {
  if (auto it = m_Prototypes.find(key); it!=end(m_Prototypes)) {
    return m_Prototypes[key]->Clone();
  }
  return nullptr;
}
Vehicle *GetRedCar() {
  auto vehicle = VehiclePrototypes::GetPrototype("car");
  vehicle->SetColor("Red");
  vehicle->SetHitPoints(10);
  vehicle->SetSpeed(30);
  vehicle->SetPosition({0,0});
  Animation anim{"red.anim"};
  vehicle->SetAnimationData(anim.GetAnimationData());
  return vehicle;
}
Vehicle *GetGreenCar() {
  auto vehicle = VehiclePrototypes::GetPrototype("car");
  vehicle->SetColor("Green");
  vehicle->SetHitPoints(5);
  vehicle->SetSpeed(30);
  vehicle->SetPosition({100,0});
  Animation anim{"green.anim"};
  vehicle->SetAnimationData(anim.GetAnimationData());
  return vehicle;
}
Vehicle* GetYellowBus() {
  auto vehicle = VehiclePrototypes::GetPrototype("bus");
  vehicle->SetColor("Yellow");
  vehicle->SetHitPoints(20);
  vehicle->SetSpeed(25);
  vehicle->SetPosition({100,200});
  Animation anim{"ybus.anim"};
  vehicle->SetAnimationData(anim.GetAnimationData());
  return vehicle;
}
Vehicle* GetBlueBus() {
  auto vehicle = VehiclePrototypes::GetPrototype("bus");
  vehicle->SetColor("Blue");
  vehicle->SetHitPoints(20);
  vehicle->SetSpeed(25);
  vehicle->SetPosition({200,200});
  Animation anim{"bbus.anim"};
  vehicle->SetAnimationData(anim.GetAnimationData());
  return vehicle;
}
void GameManager::Run() {
  m_Vehicles.push_back(GetRedCar());
  m_Vehicles.push_back(GetGreenCar());
  m_Vehicles.push_back(GetYellowBus());
  m_Vehicles.push_back(GetBlueBus());
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
int main() {
  VehiclePrototypes::RegisterPrototype("car", new Car{});
  VehiclePrototypes::RegisterPrototype("bus", new Bus{});
  GameManager mgr;
  mgr.Run();
}