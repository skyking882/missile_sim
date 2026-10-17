#include <cstring>
#include "ClogDecryptor.h"
#include "filesystem"
#include <algorithm>

namespace fs = std::filesystem;

#include "fstream"
#include "string"
#include <thread>

struct Args {
  char * path = nullptr;
  char * opath = nullptr;
  bool print = true;
  bool continuous = false;
};

std::ifstream open_clog_file(const char* path) {
  fs::path p(path);

  if (fs::is_regular_file(p))
    //std::cout << "Path is a file: " << p << std::endl;
    return std::ifstream(p, std::ios::binary);

  if (fs::is_directory(p)) {
    //std::cout << "Path is a directory: " << p << std::endl;
    fs::directory_iterator dir_iter(p), end;
    std::vector<fs::directory_entry> clog_files;
    for (auto& entry : dir_iter) {
      if (entry.is_regular_file() && entry.path().extension() == ".clog") {
        clog_files.push_back(entry);
      }
    }
    if (clog_files.empty()) {
      std::cerr << "No .clog files found in directory.\n";
      std::exit(1);
    }
    // Find most recently modified
    auto latest = std::max_element(clog_files.begin(), clog_files.end(),
                                   [](const fs::directory_entry& a, const fs::directory_entry& b) {
                                     return fs::last_write_time(a) < fs::last_write_time(b);
                                   });
    std::cout << "Most recent .clog file: " << latest->path() << std::endl;
    return std::ifstream(latest->path(), std::ios::binary);
  }
  std::cerr << "Invalid path: " << path << std::endl;
  std::exit(1);
}


void decrypt(Args * args)
{
  Decryptor decrypt{args->opath, args->print};
  std::ifstream infile = open_clog_file(args->path);
  infile.seekg(0, std::ios::end);
  std::streampos file_size = infile.tellg();
  infile.seekg(0, std::ios::beg); // Reset pointer!
  std::vector<unsigned char> buffer(file_size);
  infile.read(reinterpret_cast<char*>(buffer.data()), file_size);
  decrypt.decrypt(buffer);
}

[[noreturn]] void decrypt_continuous(Args * args)
{
  Decryptor decrypt{args->opath, args->print};
  std::ifstream infile = open_clog_file(args->path);
  std::streampos last_pos = 0;
  bool cont = true;
  while (cont)
  {
    infile.clear(); // Clear any error flags
    infile.seekg(0, std::ios::end);
    std::streampos file_size = infile.tellg();
    if (file_size > last_pos) {
      size_t read_size = static_cast<size_t>(file_size - last_pos);
      infile.seekg(last_pos, std::ios::beg);
      std::vector<unsigned char> buffer(read_size);
      infile.read(reinterpret_cast<char*>(buffer.data()), read_size);
      if (infile) {
        cont = decrypt.decrypt(buffer);
        last_pos = file_size;
      } else {
        std::cerr << "Read error.\n";
        infile.clear();
      }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
  std::exit(0);
}

void print_usage() {
  std::cout << "USAGE\n"
               "-C continuous mode, will attempt to read and decrypt new data from the file, the process must be killed"
               "-O output file, if not passed, will assume no output file"
               "-I Input file, if the path is a directory, will attempt to read the most recent .clog from that directory"
               "-P print mode, give it either true or false (case insensitive) to determine if will toString decrypted clog";
};

void exit_error(const std::string& message)
{
  std::cerr << message;
  std::exit(1);
}

void to_lowercase(char* str) {
  for (size_t i = 0; i < strlen(str); ++i) {
    str[i] = std::tolower(static_cast<unsigned char>(str[i]));
  }
}

int main(int argc, char *argv[]) {
  Args args;
  bool continuous = false;
  if (argc < 3)
  {
    print_usage();
    std::exit(0);
  }
  for (int i = 1; i < argc; i++) // first arg is exe path, so start from index 1
  {
    char * curr_arg = argv[i];
    if (curr_arg[0] == '-')
    {
      curr_arg++;
      if (strlen(curr_arg) > 1)
      {
        exit_error("An Option had length greater than 1");
      }
      switch (curr_arg[0])
      {
        case 'C':
        {
          continuous = true;
          break;
        }
        case 'O':
        {
          if (i+1 >= argc || argv[i+1][0] == '-')
            exit_error("-O missing input");
          args.opath = argv[i+1];
          i++;
          break;
        }
        case 'I':
        {
          if (i+1 >= argc || argv[i+1][0] == '-')
            exit_error("-O missing input");
          args.path = argv[i+1];
          i++;
          break;
        }
        case 'P':
        {
          if (i+1 >= argc || argv[i+1][0] == '-')
            exit_error("-P missing input");
          to_lowercase(argv[i+1]);
          char * arg = argv[i+1];
          if(strcmp(arg, "true") == 0)
          {
            args.print = true;
          }
          else if (strcmp(arg, "false") == 0)
          {
            args.print = false;
          }
          else
          {
            exit_error("-P Given argument not matching 'true' or 'false'");
          }
          i++;
          break;
        }
        default:
        {
          std::cerr << "Unsupported Argument: " << curr_arg;
          std::exit(1);
        }
      }
    }
    else
    {
      exit_error("Invalid input");
    }
  }
  if(args.path)
  {
    if(!std::filesystem::exists(args.path))
      exit_error("Input Path does not exist");
  }
  else
  {
    exit_error("Input Path not provided");
  }
  if(args.opath)
  {
    FILE * file = fopen(args.opath, "w");
    if (!file)
    {
      exit_error("Unable to write to provided output file");
    }
  }
  if(continuous)
  {
    decrypt_continuous(&args);
  }
  else
  {
    decrypt(&args);
  }
  //std::cout << "Running with options:\n";
  //std::string path = R"(D:\SteamLibrary\steamapps\common\War Thunder\.game_logs\2025_09_18_20_50_11__2884.clog)";
  //Decryptor decrypt{nullptr, true};
  //std::ifstream infile{path, std::ios::binary};
  //infile.seekg(0, std::ios::end);
  //std::streampos file_size = infile.tellg();
  //infile.seekg(0, std::ios::beg); // Reset pointer!
  //std::vector<unsigned char> buffer(file_size);
  //infile.read(reinterpret_cast<char*>(buffer.data()), file_size);
  //decrypt.decrypt(buffer);

}