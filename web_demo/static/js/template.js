

function display_image() {
        document.getElementById('show_image').innerHTML = ""
        obj = uploadFile()
        document.getElementById('show_image').innerHTML +="<img class=\"img-responsive\" src=\"static/images/"+obj.name+"\">";
  }

function httpGet(theUrl)
      {
          var xmlHttp = new XMLHttpRequest();
          xmlHttp.open( "GET", theUrl, false ); // false for synchronous request
          xmlHttp.send( null );
          var myObj = JSON.parse(xmlHttp.responseText);
          return myObj;
      }
  
function uploadFile(){
        var file = document.getElementById("image-file").files[0];
        var thresh = document.getElementById("thresh").value
        var formdata = new FormData();
        formdata.append("image-file", file);
        var ajax = new XMLHttpRequest();
        ajax.open("POST", "/upload/"+thresh, false);
        ajax.send(formdata);
        var myObj = JSON.parse(ajax.responseText);
        return myObj;
        }