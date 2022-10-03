var http = require("http");
var url = require("url");
var server = http.createServer(function(request, response) {
    var queryObj = url.parse(request.url, true).query;
    var name = queryObj.name;
    var age = queryObj.age;
    var sex = queryObj.sex;
    response.writeHead(200,{"Content-Type":"text/html;charset=UTF-8"});
    response.end("Server get request" + name + age + sex);
});

server.listen(3000,"127.0.0.1")
console.log("Server running at http://127.0.0.1:3000/")

// CommonJS demo
function loadModule(filename, module, require) {
    const wrappedSrc = 
        `(function (module, exports, require) {
            ${fs.readFileSync(filename, 'utf8')}
            })(module, module.exports, require)`
    eval(wrappedSrc)
}

// require practice
function require(moduleName) {
    console.log('Require invoked for module: ' + moduleName)
    // 1. Resolve the full module id to a file location
    const id = require.resolve(moduleName)
    // 2. If module is already cached, use its exports
    if (require.cache[id]) {
        return require.cache[id].exports
    }
    // 3. Create a new module (and put it into the cache), containing exports and id
    const module = { exports: {}, id: id }
    require.cache[id] = module
    // 4. Load the module into the module and run it
    loadModule(id, module, require)
    // 5. Return the exports of the module
    return module.exports
}

require.cache = {}
require.resolve = function (moduleName) {
    return path.resolve(moduleName)
}

// define a module in a file: myModule.js