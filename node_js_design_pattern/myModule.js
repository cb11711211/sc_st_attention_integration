// define a module
const dependency = require('./dependency')

// function
function log() {
    console.log('Well done ${dependency.name}!')
}

// export API 
module.exports.run = () => {
    log()
}

// add new content to the module
exports.hello = () => {
    console.log('Hello World')
}
// or 
module.exports = () => {
    console.log('Hello World')
}

