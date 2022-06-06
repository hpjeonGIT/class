#include "AppFrameWork.h"
#include <iostream>
void Application::New() {  m_pDocument = Create();}
void Application::Open() {
  m_pDocument = Create();
  m_pDocument->Read();
}
void Application::Save() {  m_pDocument->Write();}
DocumentPtr TextApplication::Create() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;
  return std::make_unique<TextDocument>();
}
void TextApplication::New() {  m_pDocument = Create();}
void TextApplication::Open() {
  m_pDocument = Create();
  m_pDocument->Read();
}
void TextApplication::Save() {  m_pDocument->Write();}
void TextDocument::Write() {  std::cout <<  __PRETTY_FUNCTION__  << std::endl;}
void TextDocument::Read() {  std::cout <<  __PRETTY_FUNCTION__  << std::endl;}
DocumentPtr SpreadsheetApplication::Create() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;
  return std::make_unique<SpreadsheetDocument>();
}
void SpreadsheetApplication::New() {  m_pDocument = Create();}
void SpreadsheetApplication::Open() {
  m_pDocument = Create();
  m_pDocument->Read();
}
void SpreadsheetApplication::Save() {  m_pDocument->Write();}
void SpreadsheetDocument::Write() {  std::cout <<  __PRETTY_FUNCTION__  << std::endl;}
void SpreadsheetDocument::Read() {  std::cout <<  __PRETTY_FUNCTION__  << std::endl;}
int main() {
  TextApplication app;
  app.Create();
  app.Open();
  app.Save();
  SpreadsheetApplication shapp;
  shapp.Create();
  shapp.Open();
  shapp.Save();
}