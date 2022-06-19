#pragma once
#include <iostream>
#include <string_view>
#include <vector>
#include <memory>
struct Position3D {
  int x,y,z;
  friend std::ostream& operator<< (std::ostream&os, const Position3D& obj) {
    return os << "{" << obj.x <<","<<obj.y <<","<<obj.z <<")\n";
  }
};
class Model {
public:
  virtual void Render();
  virtual void Render(Position3D position);
};
class Vegetation: public Model {
  inline static int m_Count {};
  std::vector<int> m_MeshData{};
  const char *m_Texture{};
  std::string m_Tint{};
  Position3D m_Position{};
public:
  Vegetation(std::string_view tint, Position3D position);
  void Render() override;
  static void ShowCount();
};