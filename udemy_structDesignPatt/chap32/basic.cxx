#include "basic.h"
#include <sstream>
#include <stdexcept>
#include <iostream>
#include <fstream>
Employee::Employee(const std::string& name, 
                   const std::string &role) : m_Name{name}, m_Role{role} {
                      
}
const std::string& Employee::GetName() const {
  return m_Name;  
}
const std::string& Employee::GetRole() const {
  return m_Role;
}
std::string Employee::GetInfo() const {
  std::ostringstream out;
  out << '[' << m_Role << ']' << m_Name << ' ';
  return out.str();
}
Repository::Repository(const std::string& repoPath) : m_CurrentPath{repoPath} {

}
void Repository::CreateFile(const std::string& fileName) {
  auto path = m_CurrentPath;
  path /= fileName;
  std::ofstream out{path};
  if(!out.is_open()) {
    throw std::runtime_error{"Could not create file"};    
  }
  std::cout << GetUser()->GetInfo() << " is creating a file\n";
  std::string fileData;
  std::cout << "[Create] Enter data:";
  getline(std::cin, fileData);
  out << fileData;
  std::cout << "File created successfully!";
}
void Repository::DeleteFile(const std::string& fileName) {
  auto path = m_CurrentPath;
  path /= fileName;
  if (!exists(path)) {
    throw std::runtime_error("Path does not exist");
  }
  std::cout << GetUser()->GetInfo() << " is deleting a file\n";
  if (remove(path)) {
    std::cout << "File deleted successfully";
  }  
}
void Repository::UpdateFile(const std::string& fileName) {
  auto path = m_CurrentPath;
  path /= fileName;
  std::ofstream out{path, out.app};
  if(!out.is_open()) {
    throw std::runtime_error{"Could not open file"};    
  }
  std::cout << GetUser()->GetInfo() << " is updating a file\n";
  std::string fileData;
  std::cout << "[Update] Enter data:";
  getline(std::cin, fileData);
  out << "\n### UPDATE ####\n" << fileData;
  std::cout << "File updated successfully!";
}
void Repository::ViewFile(const std::string& fileName) {
  auto path = m_CurrentPath;
  path /= fileName;
  std::ifstream in{path};
  if(!in.is_open()) {
    throw std::runtime_error{"Could not open file"};    
  }
  std::cout << GetUser()->GetInfo() << " is viewing a file\n";
  std::string line;
  while(getline(in, line)) {
    std::cout << line << std::endl;
  }
}
int main() {
  try{
    Repository repo{R"(./)"};
    std::shared_ptr<Employee> e1 (new Employee {"Umar", "Progerammer"});
    std::shared_ptr<Employee> e2 (new Employee {"Ayamm", "Manager"});
    repo.SetEmployee(e1);
    //repo.CreateFile("data.txt");
    repo.SetEmployee(e2);
    repo.ViewFile("data.txt");
  } catch (std::exception &ex) {
    std::cout <<"Exception:" << ex.what() << std::endl;
  }
}