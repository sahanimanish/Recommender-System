console.log("hello World");
buttonDOM = document.querySelector(".search-btn");
buttonDOM.addEventListener("click", (event) => {
  event.preventDefault();
  var x =document.querySelector('.search-bar').value 
  url = '/'+x
  window.location.replace(url);
  
});
