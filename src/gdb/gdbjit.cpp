#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cstdint>

#include "gdbjit.h"

// ref: https://github.dev/erlang/otp/blob/de4ff98a6e4039f16f8869ffe94cabadb8cdf985/erts/emulator/beam/jit/beam_jit_metadata.cpp
extern "C" {

typedef enum {
	JIT_NOACTION = 0,
	JIT_REGISTER_FN,
	JIT_UNREGISTER_FN,
} jit_actions_t;

struct jit_code_entry {
	struct jit_code_entry *next_entry;
	struct jit_code_entry *prev_entry;
	const char *symfile_addr;
	uint64_t symfile_size;
};

struct jit_descriptor {
	uint32_t version;
	// This type should be jit_actions_t, but we use uint32_t
	// to be explicit about the bitwidth.
	uint32_t action_flag;
	struct jit_code_entry *relevant_entry;
	struct jit_code_entry *first_entry;
};

// GDB puts a breakpoint in this function.
void __attribute__((noinline)) __jit_debug_register_code() {}

// Make sure to specify the version statically, because the
// debugger may check the version before we can set it.
struct jit_descriptor __jit_debug_descriptor = { 1, 0, 0, 0 };

} // extern "C"

void GDBJit::addDebugLine(size_t offset, const char *file_name, int line_number){
	debugEntries.emplace_back(offset, file_name, line_number);
}

struct line_info {
	char name[128];
	uint64_t addr;
	int32_t line_num;
};
void GDBJit::addCodeSegment(const char *name, uint64_t addr, uint64_t size){
	puts("register_code called");
	printf("name: %s; addr: %lu; size: %lu\n", name, addr, size);

	uint64_t name_size = strlen(name)+1; // including null terminator
	uint64_t symfile_size = name_size + 3*sizeof(uint64_t) + debugEntries.size() * sizeof(line_info);
	char *symfile = (char*)malloc(symfile_size);
	char *ptr = symfile;
	// begin address
	*(uint64_t*)ptr = addr;
	ptr += sizeof(uint64_t);
	// end address
	*(uint64_t*)ptr = addr + size;
	ptr += sizeof(uint64_t);
	// num
	*(uint64_t*)ptr = debugEntries.size();
	ptr += sizeof(uint64_t);	// function/symbol name
	(void)memcpy(ptr, name, name_size);
	ptr += name_size;
	auto p = (line_info*)ptr;
	for (size_t i = 0; i < debugEntries.size(); i++) {
		strcpy(p[i].name, debugEntries[i].file);
		p[i].line_num = debugEntries[i].line;
		p[i].addr = debugEntries[i].offset + addr;
	}

	// create entry
	jit_code_entry *n = new jit_code_entry; //FIXME: memory leak currently
	n->next_entry = nullptr;
	n->prev_entry = nullptr;
	n->symfile_addr = symfile;
	n->symfile_size = symfile_size;
	// insert into linked list
	jit_code_entry *entry = __jit_debug_descriptor.first_entry;
	n->next_entry = entry;
	if(entry){
		entry->prev_entry = n;
	}
	__jit_debug_descriptor.first_entry = n;
	// let GDB know about the new entry
	__jit_debug_descriptor.action_flag = JIT_REGISTER_FN;
	__jit_debug_descriptor.relevant_entry = n;

	puts("calling GDB");
	__jit_debug_register_code();
}
