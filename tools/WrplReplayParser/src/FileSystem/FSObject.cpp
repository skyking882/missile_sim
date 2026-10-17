
#include "FileSystem.h"

void FSObject::unBind() {
    if(!owner)
        return;
    owner->RemoveFsObject(this->getName());
}

Directory* FSObject::asDirectory() {
    if (obj_type == isDirectory)
    {
        return dynamic_cast<Directory*>(this);
    }
    return nullptr;
}

FileIndex *FSObject::asFile() {
    if (obj_type == isFile)
    {
        return dynamic_cast<FileIndex *>(this);
    }
    return nullptr;
}

SmartFSHandle SmartFSHandle::operator[](const std::string &name) const
{
    if (!ptr)
        throw std::runtime_error("null FSObject dereferenced");

    return (*ptr)[name];
}

std::shared_ptr<FileIndex> SmartFSHandle::asFile() {
    return std::dynamic_pointer_cast<FileIndex>(ptr);
}

std::shared_ptr<Directory> SmartFSHandle::asDirectory()
{
    return std::dynamic_pointer_cast<Directory>(ptr);
}



