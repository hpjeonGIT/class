#include "model.h"
void Model::Render() {

}
void Model::Render(Position3D position){}
Vegetation::Vegetation(std::string_view tint, 
                       Position3D position) 
            : m_Tint{tint}, m_Position{position} {
  ++m_Count;
  m_MeshData.assign({5,1,2,8,2,9});
  m_Texture = R"(
    #
   ###
  #####
    #
    #
    #
)";
}
void Vegetation::Render() {
  std::cout << m_Texture;
  std::cout << "Mesh: " ;
  for (auto m: m_MeshData) {
    std::cout << m << " ";    
  }
  std::cout << "\nTint" << m_Tint << std::endl;
  std::cout << "Position: " << m_Position << std::endl;
}
//void Vegetation::Render(Position3D position){}
void Vegetation::ShowCount() {
  std::cout << "Total objects created: " << m_Count << std::endl;
}
int main() {
  std::vector<std::shared_ptr<Vegetation>> m_Trees{};
  for(int i=0;i<15;++i) {
    if (i<5) {
      m_Trees.push_back(std::make_shared<Vegetation>("Green", Position3D{i*10,i*10,i*10}));
    } else if (i>5 && i <= 10) {
      m_Trees.push_back(std::make_shared<Vegetation>("Dark Green", Position3D{i*10,i*10+10,i*10}));
    } else {
      m_Trees.push_back(std::make_shared<Vegetation>("Light Green", Position3D{i*10+10,i*10,i*10}));
    }
  }
  for (auto tree: m_Trees) {
    tree->Render();
  }
  Vegetation::ShowCount();
}