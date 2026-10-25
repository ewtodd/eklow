// Python's text formatting, as Jinja2 applies it to JSON values when a chat template prints them
// ({{ x }} is str(x), | upper is str(x).upper(), | trim is str(x).strip(), dictsort compares
// str.lower() keys). Exact for what JSON carries, with one documented limit: case mapping covers
// ASCII, Latin-1, Latin Extended-A, Greek and Cyrillic; other scripts' letters are left as they are.
#pragma once

#include <string>
#include <string_view>

#include "core/json_value.hpp"

namespace eklow::text::py {

// str(x): None, True/False, ints, floats as repr, strings as themselves, lists and dicts with their
// elements as repr.
std::string str(const json::Value& v);
// repr(x); strings quoted the way Python chooses.
std::string repr(const json::Value& v);
std::string repr_string(std::string_view s);
// repr(float): the shortest digits that round-trip, positional for 1e-4 <= |x| < 1e16, else
// d.ddde+XX.
std::string float_repr(double d);

// str.strip() with Python's whitespace (str.isspace): \t\n\v\f\r, \x1c-\x1f, space, \x85, \xa0,
//  ,  - ,  ,  ,  ,  , 　.
std::string strip(std::string_view s);
std::string upper(std::string_view s);
std::string lower(std::string_view s);

}  // namespace eklow::text::py
