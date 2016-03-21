var runRunner = function(){
  var data = {
    wheelPort : 12,
    videoPort : 12,
    sessionName : null
  };
  $.ajax({
      type: "POST",
      url: '/runner',
      data: data,
      //success: success,
  });
};

var stopRunner = function(){

  $.ajax({
      type: "POST",
      url: '/stop',
      data: {}
      //success: success,
  });
};




$('#runRunnerButton').on('click', function(){
  runRunner();
});

$('#stopRunnerButton').on('click', function(){
  stopRunner();
});
