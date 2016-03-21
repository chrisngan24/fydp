// THe horizontal padding for the axis
var CHART_PADDING = 0;

Chart.types.Scatter.extend({
  name: "VideoChart",
  initialize: function(data){
    console.log('My Line chart extension');
    Chart.types.Scatter.prototype.initialize.apply(this, arguments);
    
    var canvasWidth = $('#' +  this.chart.ctx.canvas.id).width();
    var canvasHeight = $('#' +  this.chart.ctx.canvas.id).height();
    this.yLabelWidth = 0;
    CHART_PADDING = Chart.defaults.global.scaleFontSize*3;
    var horPadding = CHART_PADDING/3*2;
    this.horX = 0;
    this.yL = canvasHeight - horPadding;
    this.sentimentEvents = [];
    this.dataLength = data[0]['data'].length;
  },
  draw: function(){
    Chart.types.Scatter.prototype.draw.apply(this, arguments);
    // TODO
    var x = this.horX;
    var scale = this.scale;
    this.chart.ctx.beginPath();
    this.chart.ctx.moveTo(x, 0 );
    this.chart.ctx.strokeStyle = '#ff0000';
    //HACKY HACK
    this.chart.ctx.lineTo(x, this.yL);
    this.chart.ctx.stroke();
    this.drawBox();
  },
  updateLine: function(x){
    // Erase line and redraw it
    this.horX = x;
    this.draw();
  },
  drawBox : function(){
    var that = this;
    this.sentimentEvents.forEach(function(sEvent) {
      var startF = sEvent['startFrame'];
      var endF = sEvent['endFrame'];
      var sentiment = sEvent['sentimentGood'];
      var ctx = that.chart.ctx;
      var yU = 0;
      var canvasWidth = $('#' +  ctx.canvas.id).width();
      var xL = startF / that.dataLength * (canvasWidth - CHART_PADDING) + CHART_PADDING - 10;
      var xR = endF / that.dataLength * (canvasWidth - CHART_PADDING) + CHART_PADDING - 10;
      ctx.fillStyle = "black";
      ctx.font = "20px Arial";
      if (that.textOnBox == true){
        ctx.fillText(sEvent['eventID'], xL + 10, that.yL + 10) ;
      }
      if (sentiment){
        ctx.fillStyle = "rgba(102, 204, 0, 0.2)";
      } else{
        ctx.fillStyle = "rgba(255, 102, 102, 0.2)";
      }
      ctx.fillRect(xL,yU,xR - xL,that.yL - yU);

    });
  },
  setSentimentEvents : function(sentimentEvents, textOnBox) {
    this.sentimentEvents = sentimentEvents;
    console.log(textOnBox);
    this.textOnBox = textOnBox;
  }

});

