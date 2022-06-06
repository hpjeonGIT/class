#pragma once
#include <string>
#include <iostream>
#include <random>
#include <unordered_map>
#include <memory>
class Animation {
  std::string m_AnimationData{} ;
public:
  Animation()=default;
  Animation(std::string_view animFile);
  const std::string & GetAnimationData() const {
    return m_AnimationData;    
  }
  void SetAnimationData(const std::string &animationData)  {
    m_AnimationData = animationData;
  }
};
struct Position {
  int x;
  int y;
  friend std::ostream & operator<< (std::ostream &out, Position p) {
    return out << "(" << p.x << ',' << p.y << ')';
  }
};
class Vehicle;
using VehiclePtr = std::shared_ptr<Vehicle>;
class Vehicle {
  int m_Speed{};
  int m_HitPoints{};
  std::string m_Name{};
  Animation *m_pAnimation{};
  Position m_Position{};
  std::string m_Color{};
public:
  Vehicle();
  Vehicle(int mSpeed, int mHitPoints, const std::string& mName, 
          std::string_view animFile, const Position& mPosition,
          const std::string& mColor);
  virtual ~Vehicle();
  Vehicle(const Vehicle &other); // copy constructor
  Vehicle &operator=(const Vehicle &other);  // assign operator
  Vehicle(Vehicle &&other) noexcept; // Move constructor
  Vehicle & operator=(Vehicle &&other) noexcept; // move operator
  int GetSpeed() const {    return m_Speed;  }
  int GetHitPoints() const { return m_HitPoints; }
  const std::string& GetName() const {    return m_Name;  }
  Position GetPosition() const { return m_Position; }
  const std::string & GetAnimation() const;
  const std::string& GetColor() const { return m_Color; }
  void SetSpeed(int speed) {m_Speed = speed; }
  void SetPosition(Position position) { m_Position = position; }
  void SetName(const std::string &name) { m_Name = name; }
  void SetHitPoints(int hitPoints) { m_HitPoints = hitPoints; }
  void SetAnimationData(const std::string &animData);
  void SetColor(const std::string& color) { m_Color = color; }
  virtual void Update() = 0;
  virtual VehiclePtr Clone() = 0;
};
class Car : public Vehicle {
  using Vehicle::Vehicle;
  float m_SpeedFactor{1.5f};
  std::default_random_engine m_Engine{100};
  std::bernoulli_distribution m_Dist{.5};
public:
  void SetSpeedFactory(float factor) { m_SpeedFactor = factor;}
  void Update() override;
  VehiclePtr Clone() override;
};
class Bus : public Vehicle {
  using Vehicle::Vehicle;
  std::default_random_engine m_Engine{500};
  std::bernoulli_distribution m_Dist{0.5};
public:
  void Update() override;
  VehiclePtr Clone() override;
};
class GameManager{
  std::vector<VehiclePtr> m_Vehicles{};
public:
  void Run();
  ~GameManager() = default;
};
// Vehicle * Create(std::string_view type, int mSpeed, int mHitPoints, const std::string& mName,
//           std::string_view animFile, const Position& mPosition)  {
//   if (type == "redcar") {
//     return new Car{mSpeed, mHitPoints, mName, animFile, mPosition, "Red"};
//   } else if (type == "greencar") {
//     return new Car{mSpeed, mHitPoints, mName, animFile, mPosition, "Green"};
//   } else if (type == "yellowbus") {
//     return new Bus{mSpeed, mHitPoints, mName, animFile, mPosition, "Yellow"};
//   } else if (type == "bluebus") {
//     return new Bus{mSpeed, mHitPoints, mName, animFile, mPosition, "Blue"};
//   } 
//   return nullptr;
// }
class VehiclePrototypes {
  inline static std::unordered_map<std::string, VehiclePtr> m_Prototypes{};
  VehiclePrototypes() = default;
public:
  static std::vector<std::string> GetKeys();
  static void RegisterPrototype(const std::string &key, VehiclePtr prototype);
  static VehiclePtr DeregisterPrototype(const std::string &key);
  static VehiclePtr GetPrototype(const std::string &key);
};